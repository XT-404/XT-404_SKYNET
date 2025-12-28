import torch
import torch.nn.functional as F
import comfy.model_management as mm

# ==============================================================================
# ARCHITECTURE: XT-404 OMEGA | VISUAL SUPREMACY SUITE
# VERSION: 3.0 (ANIME TUNED - GAMMA FIX)
# ==============================================================================

class XT_Mouchard_Cafard:
    HEADER = "\033[95m[XT-MOUCHARD]\033[0m"
    @staticmethod
    def scan(tag, tensor, step_description):
        if tensor.numel() == 0: return
        if torch.isnan(tensor).any():
            print(f"{XT_Mouchard_Cafard.HEADER} 💀 {tag} | STATUS: NAN DETECTED")
            return
        if tensor.ndim >= 4:
            # On affiche juste la moyenne pour vérifier que ça ne devient pas gris (0.5+) ou noir (0.0)
            print(f"{XT_Mouchard_Cafard.HEADER} 🔍 {tag} | Avg={tensor.mean().item():.3f}")

# ==============================================================================
# NODE 1: WAN LATENT DETAILER X
# ==============================================================================
class Wan_Latent_Detailer_X:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "detail_power": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0}),
                "smart_masking": ("BOOLEAN", {"default": True}),
                "frequency_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "anti_burn_protection": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("detailed_latent",)
    FUNCTION = "enhance_latent"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def enhance_latent(self, latent, detail_power, smart_masking, frequency_boost, anti_burn_protection):
        samples = latent["samples"].clone()
        noise = torch.randn_like(samples)
        mask = torch.ones_like(samples)
        
        if smart_masking:
            dims = samples.shape
            flat_view = samples.view(dims[0]*dims[2], dims[1], dims[3], dims[4]) if len(dims)==5 else samples
            flat_mean = flat_view.mean(dim=1, keepdim=True)
            local_var = (flat_mean - F.avg_pool2d(flat_mean, 3, stride=1, padding=1)).abs()
            mask_2d = torch.clamp((local_var / (local_var.max() + 1e-6)) * 3.0, 0.0, 1.0)
            mask = mask_2d.view(dims[0], 1, dims[2], dims[3], dims[4]) if len(dims)==5 else mask_2d

        if anti_burn_protection:
            # Protection contre la saturation du latent
            mask = mask * (1.0 - torch.sigmoid((samples.abs() - 1.5) * 2.0))

        return ({"samples": samples + (noise * detail_power * frequency_boost * mask)},)

# ==============================================================================
# NODE 2: WAN TEMPORAL LOCK PRO
# ==============================================================================
class Wan_Temporal_Lock_Pro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "lock_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0}),
                "motion_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.2}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stabilize"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def stabilize(self, images, lock_strength, motion_threshold):
        if images.ndim == 5: images = images.view(-1, *images.shape[-3:])
        output = torch.zeros_like(images)
        if len(images) > 0: output[0] = images[0]
        
        for i in range(1, len(images)):
            diff = (images[i] - output[i-1]).abs().mean(dim=-1, keepdim=True)
            mask = 1.0 - torch.sigmoid((diff - motion_threshold) * 50.0)
            output[i] = torch.lerp(images[i], output[i-1], mask * lock_strength)

        XT_Mouchard_Cafard.scan("TEMPORAL LOCK", output, "Stabilized")
        return (output,)

# ==============================================================================
# NODE 3: WAN OLED DYNAMIX (GAMMA FIX - ANIME READY)
# ==============================================================================
class Wan_OLED_Dynamix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "oled_black_point": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
                "specular_pop": ("FLOAT", {"default": 1.00, "min": 1.0, "max": 1.5, "step": 0.01}),
                "shadow_recovery": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 0.5, "step": 0.01}),
                "highlight_rolloff": ("FLOAT", {"default": 0.90, "min": 0.5, "max": 1.0}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_tone_map"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def apply_tone_map(self, images, oled_black_point, specular_pop, shadow_recovery, highlight_rolloff):
        if images.ndim == 5: images = images.view(-1, *images.shape[-3:])
        img = images.clone()
        
        # 1. Black Point (Contrast Anchor)
        # On ancre le noir. Tout ce qui est sous le seuil devient 0.0 pur.
        if oled_black_point > 0:
            img = (img - oled_black_point).clamp(min=0.0) * (1.0 / (1.0 - oled_black_point))

        # 2. Shadow Recovery (GAMMA METHOD)
        # CORRECTION MAJEURE : On utilise une courbe Gamma (puissance) au lieu d'une addition.
        # Cela éclaircit les gris SANS toucher au noir pur (0 reste 0).
        # Fini le voile blanc.
        if shadow_recovery > 0:
            # shadow_recovery 0.2 -> gamma 0.9 (éclaircit légèrement)
            gamma = 1.0 - (shadow_recovery * 0.5)
            img = torch.pow(img + 1e-6, gamma)

        # 3. Specular Pop (Highlights only)
        # On booste uniquement les zones très claires (peau brillante)
        if specular_pop > 1.0:
            luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            # Masque parabolique : affecte surtout les zones > 0.7
            spec_mask = luma.clamp(0, 1).pow(3.0).unsqueeze(-1)
            img = img + ((specular_pop - 1.0) * spec_mask)

        # 4. Rolloff (Soft Clip)
        # Compression douce des blancs pour ne pas brûler les détails
        threshold = highlight_rolloff
        mask_high = (img > threshold).float()
        
        # Formule Reinhard locale
        delta = (img - threshold).clamp(min=0.0)
        # Compression plus douce :
        compressed = threshold + (delta / (1.0 + delta * 2.0)) * (1.0 - threshold)
        
        img_final = img * (1.0 - mask_high) + compressed * mask_high
        img_final = torch.clamp(img_final, 0.0, 1.0)
        
        XT_Mouchard_Cafard.scan("OLED DYNAMIX", img_final, "Tone Mapped")
        return (img_final,)

# ==============================================================================
# NODE 4: WAN ORGANIC SKIN
# ==============================================================================
class Wan_Organic_Skin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "film_iso": ("INT", {"default": 50, "min": 0, "max": 3200}),
                "resolution_aware": ("BOOLEAN", {"default": True}),
                "chroma_grain": ("BOOLEAN", {"default": False}),
                "skin_protection": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_texture"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def apply_texture(self, images, film_iso, resolution_aware, chroma_grain, skin_protection):
        if images.ndim == 5: images = images.view(-1, *images.shape[-3:])
        b, h, w, c = images.shape
        device = images.device
        
        scale = 0.5 if (resolution_aware and h*w > 2000000) else 1.0
        # Réduction de l'intensité globale pour l'anime
        intensity = (film_iso / 60000.0) * scale
        
        noise = torch.randn((b, h, w, c if chroma_grain else 1), device=device)
        if not chroma_grain: noise = noise.repeat(1, 1, 1, c)
            
        luma = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
        # Masque : Grain visible dans les tons moyens, pas dans le noir pur ni le blanc pur
        mask = 1.0 - (2.0 * luma.unsqueeze(-1) - 1.0).pow(4.0) 
        
        if skin_protection:
            skin = (images[..., 0] > images[..., 1]) & (images[..., 1] > images[..., 2])
            mask = mask * (1.0 - (skin.float().unsqueeze(-1) * 0.5))

        final = torch.clamp(images + (noise * intensity * mask), 0.0, 1.0)
        XT_Mouchard_Cafard.scan("ORGANIC SKIN", final, "Textured")
        return (final,)

NODE_CLASS_MAPPINGS = {
    "Wan_Latent_Detailer_X": Wan_Latent_Detailer_X,
    "Wan_Temporal_Lock_Pro": Wan_Temporal_Lock_Pro,
    "Wan_OLED_Dynamix": Wan_OLED_Dynamix,
    "Wan_Organic_Skin": Wan_Organic_Skin
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Latent_Detailer_X": "1. Wan Latent Detailer X (Anti-Clip)",
    "Wan_Temporal_Lock_Pro": "2. Wan Temporal Lock Pro (Monitor)",
    "Wan_OLED_Dynamix": "3. Wan OLED Dynamix (ARRI Rolloff)",
    "Wan_Organic_Skin": "4. Wan Organic Skin (Optical Blend)"
}
