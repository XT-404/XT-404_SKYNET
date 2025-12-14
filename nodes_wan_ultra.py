import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import time
import sys
import gc

class WanImageToVideoUltra:
    """
    WanImageToVideoUltra - Version DEFINITIVE (Fidelity + Dynamics + Frame Control).
    Avec système de monitoring 'Mouchard' pour analyse de performance.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 114, "min": 1, "max": 4096, "step": 1, "tooltip": "Nombre exact de frames."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "detail_boost": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
                "force_ref": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "WanVideo/Ultra"

    def _log(self, step_name):
        """Mouchard Cafard: Affiche les stats mémoire et temps dans la CMD"""
        # Si c'est l'étape d'initialisation
        if step_name == "Init":
            self.t0 = time.time()
            self.step_t0 = self.t0
            # flush=True force l'écriture immédiate dans la console Windows
            print(f"\n--- [WanUltra] DEMARRAGE DU TRAITEMENT ---", flush=True)
            return
        
        # Pour les autres étapes, on calcule le delta
        current_time = time.time()
        dt = current_time - self.step_t0
        total_t = current_time - self.t0
        
        # Stats VRAM
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3) # GB
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3) # GB
        
        print(f"--- [WanUltra] {step_name} | Step: {dt:.2f}s | Total: {total_t:.2f}s | VRAM: {mem_alloc:.2f}GB (Res: {mem_reserved:.2f}GB)", flush=True)
        self.step_t0 = current_time

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                            dtype=image_tensor.dtype, device=image_tensor.device).view(1, 1, 3, 3)
        b, c, h, w = image_tensor.shape
        enhanced = torch.nn.functional.conv2d(image_tensor.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        return torch.lerp(image_tensor, enhanced, factor * 0.3).clamp(0.0, 1.0)

    def execute(self, positive, negative, vae, video_frames, width, height, batch_size, detail_boost, motion_amp, force_ref, start_image=None, clip_vision_output=None):
        # 0. Initialisation Mouchard
        self._log("Init")
        
        device = comfy.model_management.intermediate_device()
        
        # Nettoyage préventif
        gc.collect()
        comfy.model_management.soft_empty_cache()
        self._log("Nettoyage Cache Initial")

        length = video_frames
        latent_t = ((length - 1) // 4) + 1
        
        # Allocation Latent vide (faible coût)
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=device, dtype=torch.float32)

        if start_image is not None:
            # 1. Préparation Image
            img_tensor = start_image.movedim(-1, 1).to(device=device, dtype=torch.float32)
            
            if img_tensor.shape[2] != height or img_tensor.shape[3] != width:
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=(height, width), mode="bicubic", antialias=True)
            
            if detail_boost > 0:
                img_tensor = self._enhance_details(img_tensor, detail_boost)
            
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            self._log("Preparation Image HD")

            # 2. Reference Encoding
            ref_latent = None
            if force_ref:
                # Encodage Image Seule (Léger)
                ref_latent = vae.encode(img_tensor.movedim(1, -1))
                self._log("Encodage Reference (force_ref)")
            
            # 3. Préparation Volume Vidéo
            img_tensor = img_tensor.movedim(1, -1)
            valid_frames = min(img_tensor.shape[0], length)
            
            # Création du tenseur vidéo complet
            video_input = torch.full((length, height, width, 3), 0.5, dtype=torch.float32, device=device)
            video_input[:valid_frames] = img_tensor[:valid_frames]
            
            # Nettoyage immédiat des tenseurs inutiles avant la grosse charge VAE
            del img_tensor, start_image
            self._log("Creation Volume Video")

            # 4. Encodage VAE (LE POINT CRITIQUE)
            comfy.model_management.soft_empty_cache()
            
            try:
                concat_latent_image = vae.encode(video_input)
            except Exception as e:
                print(f"!!! ERREUR VAE ENCODE !!! Manque de VRAM probable. {e}", flush=True)
                raise e
                
            del video_input
            comfy.model_management.soft_empty_cache() 
            self._log("Encodage VAE Video (Grosse Charge)")

            # 5. Motion Amplification
            if motion_amp > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                scaled_latent = base_latent + diff_centered * motion_amp + diff_mean
                
                scaled_latent = torch.clamp(scaled_latent, -6.0, 6.0)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                self._log("Application Motion Amp")

            # 6. Masquage et Conditioning
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=device, dtype=torch.float32)
            start_latent_end_index = ((valid_frames - 1) // 4) + 1
            mask[:, :, :start_latent_end_index] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

            if ref_latent is not None:
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            del mask, concat_latent_image, ref_latent
            self._log("Finalisation Conditioning")

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        
        # Nettoyage Final et rapport
        gc.collect()
        comfy.model_management.soft_empty_cache()
        self._log("Fin & Nettoyage Final")
        
        return (positive, negative, out_latent)
