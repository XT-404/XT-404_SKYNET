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
    WanImageToVideoUltra - OMEGA FIX V13 (Ultimate + Mouchard + VRAM FIX).
    Combine :
    1. La stabilité absolue de la V12 (Nuclear Normalization + Soft Limiter).
    2. Le retour du "Mouchard" complet pour le monitoring VRAM/Temps.
    3. CORRECTION VRAM : Retour à l'encodage référence simple (fix du repeat x9).
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
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
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
        if step_name == "Init":
            self.t0 = time.time()
            self.step_t0 = self.t0
            print(f"\n--- [WanUltra] DEMARRAGE DU TRAITEMENT ---", flush=True)
            return
        
        current_time = time.time()
        dt = current_time - self.step_t0
        total_t = current_time - self.t0
        
        # Stats VRAM
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3) # GB
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3) # GB
        
        print(f"--- [WanUltra] {step_name} | Step: {dt:.2f}s | Total: {total_t:.2f}s | VRAM: {mem_alloc:.2f}GB (Res: {mem_reserved:.2f}GB)", flush=True)
        self.step_t0 = current_time

    def _sanitize_tensor(self, tensor, target_w, target_h):
        """Nettoyage nucléaire des dimensions + Clamping préventif"""
        if tensor.dim() < 3: raise ValueError(f"Dim too low: {tensor.shape}")

        # Correction BCHW / BHWC
        if tensor.shape[-1] != 3:
            if tensor.shape[1] == 3: tensor = tensor.movedim(1, -1)
            elif tensor.shape[0] == 3: tensor = tensor.movedim(0, -1)
                
        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
            
        # Resize safe
        tensor_bchw = tensor.movedim(-1, 1)
        tensor_resized = torch.nn.functional.interpolate(
            tensor_bchw, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
        )
        
        # OVERSHOOT FIX : Bicubic peut créer des valeurs < 0 (noir saturé). On coupe ici.
        tensor_resized = torch.clamp(tensor_resized, 0.0, 1.0)
        
        return tensor_resized.movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        device = image_tensor.device
        img_bchw = image_tensor.movedim(-1, 1)
        
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                            dtype=image_tensor.dtype, device=device).view(1, 1, 3, 3)
        
        b, c, h, w = img_bchw.shape
        enhanced = torch.nn.functional.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        
        result_bchw = torch.lerp(img_bchw, enhanced, factor * 0.2)
        # Clamping post-sharpening
        result_bchw = torch.clamp(result_bchw, 0.0, 1.0)
        
        return result_bchw.movedim(1, -1)

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
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=device, dtype=torch.float32)

        if start_image is not None:
            # 1. NUCLEAR SANITIZATION
            img_final = self._sanitize_tensor(start_image, width, height).to(device, dtype=torch.float32)
            
            # 2. Detail Boost
            if detail_boost > 0:
                img_final = self._enhance_details(img_final, detail_boost)
            
            self._log(f"Preparation Image HD (Nuclear Norm)")

            # 3. Encodage Reference (CORRIGÉ - VRAM OPTIMIZED)
            ref_latent = None
            if force_ref:
                try:
                    # CORRECTION: On encode l'image unique directement au lieu de la dupliquer 9 fois.
                    # img_final a déjà la forme (1, H, W, 3) grace a _sanitize_tensor.
                    # Le VAE WanVideo gère très bien l'encodage d'une frame unique pour la ref.
                    ref_latent = vae.encode(img_final)
                    
                    # On s'assure qu'on a bien les dimensions attendues (par sécurité)
                    if ref_latent.shape[2] > 1:
                         ref_latent = ref_latent[:, :, 0:1, :, :]
                         
                    self._log("Encodage Reference (Optimized)")
                except Exception as e:
                    print(f"!!! ERREUR REF ENCODE (Skip): {e}", flush=True)
                    ref_latent = None
            
            # 4. Préparation Volume Vidéo
            valid_frames = min(img_final.shape[0], length)
            video_input = torch.full((length, height, width, 3), 0.5, dtype=torch.float32, device=device)
            video_input[:valid_frames] = img_final[:valid_frames]
            
            del start_image
            self._log("Creation Volume Video")

            # 5. Encodage VAE Principal
            comfy.model_management.soft_empty_cache()
            try:
                concat_latent_image = vae.encode(video_input)
            except Exception as e:
                print(f"!!! ERREUR VAE PRINCIPAL: {e}", flush=True)
                raise e
            
            del video_input, img_final
            self._log("Encodage VAE Video (Grosse Charge)")

            # 6. MOTION AMPLIFICATION (SOFT LIMITER UPGRADE)
            if motion_amp > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                
                # Normalisation
                std = diff.std()
                if std > 1.0: diff = diff / std
                
                # Soft Limiter
                limit = 2.5 
                boosted = diff * motion_amp
                soft_diff = torch.tanh(boosted / limit) * limit
                
                scaled_latent = base_latent + soft_diff
                
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                self._log("Application Motion Amp (Soft Limiter)")

            # 7. Conditioning
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

        out_latent = {"samples": latent}
        
        # Nettoyage Final et rapport
        gc.collect()
        comfy.model_management.soft_empty_cache()
        self._log("Fin & Nettoyage Final")
        
        return (positive, negative, out_latent)
