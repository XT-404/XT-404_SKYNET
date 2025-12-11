import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils

class WanImageToVideoFidelity:
    """
    Version 'Haute Fidélité' optimisée.
    - Restaure la logique originale (encodage complet) pour une cohérence parfaite (contexte temporel).
    - Force le FP32 et l'interpolation Bicubic pour une qualité d'image maximale.
    - Optimise l'allocation mémoire (torch.full) et nettoie la VRAM agressivement pour éviter le lag.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "WanVideo/Optimized"

    def execute(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        device = comfy.model_management.intermediate_device()
        
        # Calcul de la dimension temporelle latente (Formule Wan2.1)
        latent_t = ((length - 1) // 4) + 1
        
        # Initialisation du latent de sortie
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=device)

        if start_image is not None:
            # 1. PRÉPARATION DE L'IMAGE (Qualité Maximale)
            # - Upscale "bicubic" pour préserver les détails fins (mieux que bilinear).
            # - .float() force le FP32 pour éviter la perte de précision (banding).
            start_image_upscaled = comfy.utils.common_upscale(
                start_image.movedim(-1, 1), width, height, "bicubic", "center"
            ).movedim(1, -1).float()
            
            # On s'assure de ne pas dépasser la longueur demandée
            valid_frames = min(len(start_image_upscaled), length)
            current_start = start_image_upscaled[:valid_frames]

            # 2. CRÉATION DU CANVAS (Optimisation Mémoire)
            # Au lieu de 'torch.ones * 0.5', on utilise 'torch.full' qui est plus rapide et direct.
            # On crée le tenseur complet ici pour que le VAE puisse voir le contexte temporel (transition image -> gris).
            # C'est ce qui restaure la cohérence par rapport à la version précédente.
            video_input = torch.full(
                (length, height, width, 3), 
                0.5, 
                dtype=current_start.dtype, 
                device=current_start.device
            )
            
            # On insère l'image de départ
            video_input[:valid_frames] = current_start
            
            # CRITIQUE : On supprime immédiatement l'image source pour libérer la VRAM avant le gros calcul VAE
            del start_image_upscaled, current_start, start_image

            # 3. ENCODAGE VAE (Optimisé)
            # On passe tout le bloc au VAE. C'est l'opération lourde, mais nécessaire pour la qualité.
            # ComfyUI gère généralement le tiling spatial, ce qui évite le OOM si configuré.
            concat_latent_image = vae.encode(video_input)
            
            # Nettoyage immédiat du gros tenseur pixel (Libère ~1Go+ de VRAM)
            del video_input

            # 4. CRÉATION DU MASQUE (Logique Originale)
            # On recrée exactement le masque dont le modèle a besoin.
            # On le met directement en FP32.
            mask = torch.ones(
                (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), 
                device=device, 
                dtype=torch.float32
            )
            
            # Calcul de l'index de fin dans l'espace latent
            start_latent_end_index = ((valid_frames - 1) // 4) + 1
            mask[:, :, :start_latent_end_index] = 0.0

            # 5. INJECTION DU CONDITIONING
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            
            # Nettoyage final
            del mask, concat_latent_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)