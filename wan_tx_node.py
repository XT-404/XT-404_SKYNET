import torch
import torch.nn.functional as F
import nodes
import node_helpers
import comfy.model_management as mm
import gc
import time

class Wan_TX_Fusion:
    """
    CYBERDYNE SYSTEMS: T-X FUSION (v3.2 Stable)
    HYBRID CORE: Safety Architecture + Advanced Motion Engine (ISR)
    Code updated for CPU/GPU mixed context stability.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 81, "min": 5, "max": 4096, "step": 4, "tooltip": "Taille temporelle (4n+1 recommandé pour Wan)."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "start_image": ("IMAGE", ),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Amélioration de la netteté de l'image source."}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Amplification du mouvement via séparation de fréquences."}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp32", "tooltip": "Précision des calculs."}),
                "tile_strategy": (["Default (Fast)", "Tiled (VRAM Saver)"], {"default": "Default (Fast)"}),
            },
            "optional": {
                "end_image": ("IMAGE", ),
                "clip_vision_start": ("CLIP_VISION_OUTPUT", ),
                "clip_vision_end": ("CLIP_VISION_OUTPUT", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "XT-404/Wan2.2"

    def _log(self, step_name, info=""):
        if step_name == "Init":
            self.t0 = time.time()
            print(f"\n[T-X FUSION] 🚀 Initialisation...", flush=True)
            return
        elapsed = time.time() - self.t0
        print(f"👉 [T-X] {step_name:<15} | {info} ({elapsed:.2f}s)", flush=True)

    def _get_dtype(self, precision_str):
        if precision_str == "fp16": return torch.float16
        if precision_str == "bf16": return torch.bfloat16
        return torch.float32

    def _sanitize_tensor(self, tensor, target_w, target_h, device, dtype):
        """Redimensionne et nettoie les tenseurs images."""
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype)
        
        # Gestion batch
        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        
        # Interpolation (B, H, W, C) -> (B, C, H, W) -> Resize -> (B, H, W, C)
        tensor_bchw = tensor.movedim(-1, 1)
        if tensor_bchw.shape[2] != target_h or tensor_bchw.shape[3] != target_w:
            tensor_resized = F.interpolate(tensor_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            tensor_resized = tensor_bchw
            
        return torch.clamp(tensor_resized, 0.0, 1.0).movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        """Ajoute un micro-contraste pour aider le modèle à garder les détails."""
        if factor <= 0: return image_tensor
        img_bchw = image_tensor.movedim(-1, 1)
        # Filtre de sharpening simple
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                              dtype=image_tensor.dtype, device=image_tensor.device).view(1, 1, 3, 3)
        b, c, h, w = img_bchw.shape
        # Appliquer par canal
        enhanced = F.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        # Mélange : Original + (Enhanced - Original)*Factor
        return torch.lerp(img_bchw, enhanced, factor * 0.2).clamp(0,1).movedim(1, -1)

    def execute(self, positive, negative, vae, video_frames, width, height, start_image, detail_boost, motion_amp, precision, tile_strategy, end_image=None, clip_vision_start=None, clip_vision_end=None):
        self._log("Init")
        device = mm.get_torch_device()
        target_dtype = self._get_dtype(precision)
        
        # 1. Préparation des Images (Start / End)
        resized_start = self._sanitize_tensor(start_image, width, height, device, target_dtype)
        if detail_boost > 0: resized_start = self._enhance_details(resized_start, detail_boost)

        # Création du volume vidéo (Rempli de gris neutre 0.5)
        volume = torch.full((video_frames, height, width, 3), 0.5, device=device, dtype=target_dtype)
        
        # Injection Start
        volume[0:1] = resized_start[0:1] # Frame 0
        
        valid_end = False
        if end_image is not None:
            resized_end = self._sanitize_tensor(end_image, width, height, device, target_dtype)
            if detail_boost > 0: resized_end = self._enhance_details(resized_end, detail_boost)
            volume[-1:] = resized_end[-1:] # Frame Finale
            valid_end = True

        self._log("Volume Build", f"{video_frames} frames ({width}x{height})")

        # 2. Encodage VAE (Méthode Safe + Gestion VRAM)
        try:
            if "Tiled" in tile_strategy:
                latent = vae.encode_tiled(volume[:, :, :, :3], tile_size_xy=512) 
            else:
                latent = vae.encode(volume[:, :, :, :3])
        except Exception as e:
            print(f"[T-X WARNING] VAE Encode error: {e}. Switching to CPU fallback/Tiling.")
            mm.soft_empty_cache()
            latent = vae.encode_tiled(volume[:, :, :, :3])

        self._log("Encoding", f"Latent Shape: {list(latent.shape)}")

        # 3. Moteur ISR (Inverse Structural Repulsion)
        final_latent = latent
        if valid_end and motion_amp > 1.001 and video_frames > 4:
            # Trajectoire linéaire
            start_l = latent[:, :, 0:1]
            end_l = latent[:, :, -1:]
            
            # --- CORRECTIF APPLIQUÉ ICI ---
            # Utilisation de latent.device pour s'assurer que t_steps est sur le même device que le latent (CPU ou GPU)
            t_steps = torch.linspace(0.0, 1.0, latent.shape[2], device=latent.device, dtype=latent.dtype).view(1, 1, -1, 1, 1)
            # -----------------------------
            
            linear_latent = start_l * (1 - t_steps) + end_l * t_steps

            # Différence
            diff = latent - linear_latent
            
            # Séparation de Fréquences
            h_lat, w_lat = diff.shape[-2], diff.shape[-1]
            low_freq = F.interpolate(diff.view(-1, 16, h_lat, w_lat), size=(h_lat // 8, w_lat // 8), mode='area')
            low_freq = F.interpolate(low_freq, size=(h_lat, w_lat), mode='bilinear').view_as(diff)
            high_freq = diff - low_freq 

            # Boost
            boost_scale = (motion_amp - 1.0) * 2.0
            final_latent = latent + (high_freq * boost_scale * 1.5) + (low_freq * boost_scale * 0.3)
            self._log("Motion Engine", f"ISR Active | Amp: {motion_amp}x")

        # 4. Système de Masquage Temporel Robuste (CORRECTIF 5D)
        target_t = final_latent.shape[2]
        
        # Création du masque en 4D d'abord [1, T, H, W]
        # --- CORRECTIF APPLIQUÉ ICI AUSSI ---
        # Utilisation de final_latent.device pour garantir que le masque est compatible
        mask = torch.ones((1, target_t, height // 8, width // 8), device=final_latent.device, dtype=target_dtype)
        # -----------------------------------
        
        # Verrouillage Frame 0
        mask[:, 0] = 0.0 
        
        # Verrouillage Frame Finale
        if valid_end:
            mask[:, -1] = 0.0

        # CORRECTION MAJEURE : Passage en 5D [Batch, Channel, Time, Height, Width]
        # Wan Latent est 5D, le masque doit avoir 5 dimensions pour être concaténé.
        mask_final = mask.unsqueeze(1) 
        
        # Shape attendue : [1, 1, T, H, W]
        
        # 5. Gestion Clip Vision
        cv_out = None
        if clip_vision_start is not None:
            cv_out = clip_vision_start
            if clip_vision_end is not None:
                try:
                    states = torch.cat([cv_out.penultimate_hidden_states, clip_vision_end.penultimate_hidden_states], dim=-2)
                    cv_out = comfy.clip_vision.Output()
                    cv_out.penultimate_hidden_states = states
                except:
                    pass

        # 6. Packaging Final
        pos_final = node_helpers.conditioning_set_values(positive, {"concat_latent_image": final_latent, "concat_mask": mask_final})
        neg_final = node_helpers.conditioning_set_values(negative, {"concat_latent_image": final_latent, "concat_mask": mask_final})

        if cv_out is not None:
            pos_final = node_helpers.conditioning_set_values(pos_final, {"clip_vision_output": cv_out})
            neg_final = node_helpers.conditioning_set_values(neg_final, {"clip_vision_output": cv_out})

        mm.soft_empty_cache()
        gc.collect()
        self._log("Complete", "Ready for sampling.")
        
        return (pos_final, neg_final, {"samples": torch.zeros_like(final_latent)})

# Mappings pour ComfyUI
NODE_CLASS_MAPPINGS = {"Wan_TX_Fusion": Wan_TX_Fusion}
NODE_DISPLAY_NAME_MAPPINGS = {"Wan_TX_Fusion": "T-X Fusion (Wan2.1 Ultimate)"}
