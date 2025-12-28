import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
import sys

class Wan_Infiltration_Upscaler:
    """
    MODULE C : INFILTRATION VAE UPSCALER (V5.1 AUTO-LEVELS)
    - Force FP32 pour éviter le "White Cast".
    - Auto-Levels : Recale dynamiquement le contraste si l'image est délavée.
    - Noise Masking amélioré.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1}),
                "texture_noise": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.005}),
                "fix_milky_black": ("BOOLEAN", {"default": True, "tooltip": "Force le recalage des noirs (Anti-Voile)."}),
                "fix_hot_white": ("BOOLEAN", {"default": True, "tooltip": "Empêche les blancs de brûler."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "infiltrate_upscale"
    CATEGORY = "XT-404/V2_Omega"

    def infiltrate_upscale(self, latent, vae, upscale_by, texture_noise, fix_milky_black, fix_hot_white):
        samples = latent["samples"]
        device = mm.get_torch_device()
        
        # 1. DECODAGE
        vae.patcher.model.to(device)
        decoded = vae.decode(samples.to(device))
        
        # 2. FP32 FORCE & FLATTEN
        if decoded.ndim == 5:
            B, T, H, W, C = decoded.shape
            flat = decoded.view(-1, H, W, C).permute(0, 3, 1, 2).float()
        else:
            flat = decoded.permute(0, 3, 1, 2).float()

        # 3. INTERPOLATION
        up = F.interpolate(flat, scale_factor=upscale_by, mode="bicubic", align_corners=False, antialias=True)
        
        # 4. AUTO-LEVELS (LE SAUVEUR)
        # On analyse l'histogramme de l'image upscalée
        if fix_milky_black:
            # On cherche le point noir actuel (le 1er percentile)
            # Si l'image est délavée, min_val sera > 0.0 (ex: 0.1)
            b, c, h, w = up.shape
            pixels = up.view(b, c, -1)
            min_val = torch.kthvalue(pixels, int(h*w*0.01), dim=2).values.view(b, c, 1, 1)
            
            # On recale : (pixel - min) / (1 - min)
            # Cela remet le point le plus sombre à 0.0
            scale = 1.0 / (1.0 - min_val + 1e-6)
            up = (up - min_val).clamp(min=0.0) * scale

        if fix_hot_white:
            # On cherche le point blanc (99ème percentile)
            b, c, h, w = up.shape
            pixels = up.view(b, c, -1)
            max_val = torch.kthvalue(pixels, int(h*w*0.99), dim=2).values.view(b, c, 1, 1)
            
            # Si le blanc est trop bas (gris), on l'étend. S'il est > 1.0, on le ramène.
            # Ici on s'assure surtout qu'il ne dépasse pas 1.0 bêtement
            up = torch.clamp(up, max=1.0) # Simple clamp suffisant après le black fix

        # 5. NOISE
        if texture_noise > 0:
            noise = torch.randn_like(up) * texture_noise
            # Masque luma (Pas de bruit dans le noir pur qu'on vient de fixer)
            noise_mask = up * (1.0 - up) * 4.0
            up = up + (noise * noise_mask)

        # 6. SORTIE
        up = torch.clamp(up, 0.0, 1.0)
        out = up.permute(0, 2, 3, 1) # [N, H, W, C]
        
        return (out.cpu(),)

NODE_CLASS_MAPPINGS = {"Wan_Infiltration_Upscaler": Wan_Infiltration_Upscaler}
