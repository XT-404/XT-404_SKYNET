import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils

class WanImageToVideoUltra:
    """
    WanImageToVideoUltra - Version DEFINITIVE (Fidelity + Dynamics + Duration Control).
    
    Fusionne le meilleur de l'ingénierie haute performance et les astuces de la communauté :
    1. Base Ultra : FP32, Bicubic AA, Detail Boost, Memory Safe.
    2. Duration Preset : Sélection simplifiée par durée (5s, 10s, 15s...) avec mapping de frames précis.
    3. Reference Injection : Force le modèle à respecter l'identité de l'image.
    4. Motion Amplification : Force le mouvement si le modèle est trop statique.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                
                # --- MODIFICATION DURATION ---
                # Remplacement du Slider INT par un Dropdown précis
                "video_duration": (
                    [
                        "5s (114 frames)", 
                        "10s (229 frames)", 
                        "15s (342 frames)", 
                        "20s (456 frames)", 
                        "25s (570 frames)", 
                        "30s (684 frames)"
                    ], 
                    {"default": "5s (114 frames)"}
                ),
                
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                
                # --- PARAMÈTRES AVANCÉS ---
                "detail_boost": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Compense le flou du VAE. 0.5 recommandé."}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Force le mouvement. Mettre 1.1 ou 1.2 si la vidéo est trop statique."}),
                "force_ref": ("BOOLEAN", {"default": True, "tooltip": "Injecte l'image comme référence forte. Améliore drastiquement la fidélité du sujet."}),
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

    def _enhance_details(self, image_tensor, factor=0.5):
        """Filtre de netteté GPU optimisé"""
        if factor <= 0: return image_tensor
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                            dtype=image_tensor.dtype, device=image_tensor.device).view(1, 1, 3, 3)
        b, c, h, w = image_tensor.shape
        enhanced = torch.nn.functional.conv2d(image_tensor.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        return torch.lerp(image_tensor, enhanced, factor * 0.3).clamp(0.0, 1.0)

    def execute(self, positive, negative, vae, video_duration, width, height, batch_size, detail_boost, motion_amp, force_ref, start_image=None, clip_vision_output=None):
        device = comfy.model_management.intermediate_device()
        comfy.model_management.soft_empty_cache()
        
        # --- MAPPING AUTOMATIQUE DURATION -> FRAMES ---
        duration_map = {
            "5s (114 frames)": 114,
            "10s (229 frames)": 229,
            "15s (342 frames)": 342,
            "20s (456 frames)": 456,
            "25s (570 frames)": 570,
            "30s (684 frames)": 684
        }
        length = duration_map.get(video_duration, 114) # Sécurité par défaut
        
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=device, dtype=torch.float32)

        if start_image is not None:
            # 1. TRAITEMENT IMAGE HD (FP32 + Bicubic + Sharpen)
            # On traite l'image une seule fois proprement
            img_tensor = start_image.movedim(-1, 1).to(device=device, dtype=torch.float32)
            
            if img_tensor.shape[2] != height or img_tensor.shape[3] != width:
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=(height, width), mode="bicubic", antialias=True)
            
            if detail_boost > 0:
                img_tensor = self._enhance_details(img_tensor, detail_boost)
            
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            # --- FEATURE 1 : Reference Injection (Fidélité Identité) ---
            # On garde une copie encodée pure pour la référence globale
            ref_latent = None
            if force_ref:
                # On encode juste l'image seule pour la référence
                ref_latent = vae.encode(img_tensor.movedim(1, -1))
            
            # Préparation du volume vidéo
            img_tensor = img_tensor.movedim(1, -1) # (B, H, W, C)
            valid_frames = min(img_tensor.shape[0], length)
            
            video_input = torch.full((length, height, width, 3), 0.5, dtype=torch.float32, device=device)
            video_input[:valid_frames] = img_tensor[:valid_frames]
            del img_tensor, start_image

            # Encodage Principal
            concat_latent_image = vae.encode(video_input)
            del video_input
            comfy.model_management.soft_empty_cache()

            # --- FEATURE 2 : Motion Amplification (Dynamisme) ---
            # Si demandé, on applique le hack mathématique sur les latents
            if motion_amp > 1.0:
                # Séparation : Première frame vs Reste (frames grises/futures)
                base_latent = concat_latent_image[:, :, 0:1] # La frame image
                gray_latent = concat_latent_image[:, :, 1:]  # Les frames vides
                
                # Calcul du delta et amplification
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                # On "pousse" les frames vides plus loin de l'image de départ pour forcer le changement
                scaled_latent = base_latent + diff_centered * motion_amp + diff_mean
                
                # Recombinaison
                scaled_latent = torch.clamp(scaled_latent, -6.0, 6.0) # Sécurité VAE
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)

            # Masquage
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=device, dtype=torch.float32)
            start_latent_end_index = ((valid_frames - 1) // 4) + 1
            mask[:, :, :start_latent_end_index] = 0.0

            # Injection Conditioning
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

            # Injection Reference (Si activé)
            if ref_latent is not None:
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
                # Astuce Painter: Zero-latent pour le négatif sur la référence aide au contraste
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            del mask, concat_latent_image, ref_latent

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)