import torch
import torch.nn.functional as F
import comfy.model_management as mm
import time

# ==============================================================================
# ARCHITECTURE: XT-404 OMEGA | VISUAL SUPREMACY SUITE
# VERSION: 2.1 (ARRI ROLLOFF EDITION - ANTI-CLIPPING FINAL FIX)
# MODULES: LatentDetailerX -> TemporalLock -> OLEDDynamix -> OrganicSkin
# STATUS: DIAGNOSTIC MODE ACTIVE (XT-MOUCHARD)
# ==============================================================================

class XT_Mouchard_Cafard:
    """
    Système de Télémétrie Forensique.
    Analyse l'intégrité du signal pour détecter la saturation (Clipping) et les anomalies.
    """
    HEADER = "\033[95m[XT-MOUCHARD]\033[0m"
    RESET = "\033[0m"
    
    @staticmethod
    def scan(tag, tensor, step_description):
        # Calculs statistiques sur GPU pour vitesse
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Détection de saturation (Clipping)
        # On compte les pixels collés à 0.0 ou 1.0 (seuil de tolérance 1e-4)
        if tensor.ndim == 4: # Image (B, H, W, C)
            total_pixels = tensor.numel()
            clipped_low = (tensor <= 0.0001).sum().item()
            clipped_high = (tensor >= 0.9999).sum().item()
            sat_percentage = ((clipped_low + clipped_high) / total_pixels) * 100.0
            
            status_color = "\033[92m" # Vert
            if sat_percentage > 1.0: status_color = "\033[93m" # Jaune
            if sat_percentage > 5.0: status_color = "\033[91m" # Rouge (CRITIQUE)
            
            print(f"{XT_Mouchard_Cafard.HEADER} 🔍 {tag} | {step_description}")
            print(f"   └── Stats: Avg={mean:.3f} | Std={std:.3f} | Range=[{min_val:.3f}, {max_val:.3f}]")
            print(f"   └── Health: {status_color}Saturation Impact: {sat_percentage:.2f}%{XT_Mouchard_Cafard.RESET} (Low:{clipped_low} / High:{clipped_high})")
            
            if sat_percentage > 5.0:
                 print(f"   ⚠️ \033[91mALERTE: Saturation élevée détectée. Le Rolloff va devoir travailler dur !\033[0m")
        
        else: # Latent
            print(f"{XT_Mouchard_Cafard.HEADER} 🧠 {tag} | {step_description}")
            print(f"   └── Latent Stats: Avg={mean:.3f} | Std={std:.3f} | Range=[{min_val:.3f}, {max_val:.3f}]")

# ==============================================================================
# NODE 1: WAN LATENT DETAILER X (V2 - SAFE INJECTION)
# ==============================================================================

class Wan_Latent_Detailer_X:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "detail_power": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smart_masking": ("BOOLEAN", {"default": True}),
                "frequency_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "anti_burn_protection": ("BOOLEAN", {"default": True, "tooltip": "Empêche le bruit de saturer les valeurs extrêmes du latent."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("detailed_latent",)
    FUNCTION = "enhance_latent"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def enhance_latent(self, latent, detail_power, smart_masking, frequency_boost, anti_burn_protection):
        samples = latent["samples"].clone()
        device = samples.device
        
        XT_Mouchard_Cafard.scan("PHASE 1 (INPUT)", samples, "Latent Brut avant injection")

        # Génération du bruit (Haute Fréquence)
        noise = torch.randn_like(samples, device=device)
        
        # Masque de protection (Smart Masking)
        mask = torch.ones_like(samples)
        if smart_masking:
            # Détection de texture via écart-type local (variance)
            flat = samples.mean(dim=1, keepdim=True)
            # Unfold pour calculer la variance locale 3x3
            local_mean = F.avg_pool2d(flat.view(-1, 1, samples.shape[-2], samples.shape[-1]), 3, stride=1, padding=1)
            local_var = (flat.view(-1, 1, samples.shape[-2], samples.shape[-1]) - local_mean).abs()
            
            # Normalisation du masque (0 = plat, 1 = complexe)
            mask = local_var / (local_var.max() + 1e-6)
            mask = torch.clamp(mask * 3.0, 0.0, 1.0) # Boost contraste du masque
            mask = mask.view(samples.shape[0], 1, samples.shape[2], samples.shape[3], samples.shape[4])
        
        # PROTECTION ANTI-BURN (Latent Safe-Guard)
        if anti_burn_protection:
            # Calcul de la distance au zéro (intensité du signal)
            signal_mag = samples.abs()
            # Si le signal est > 1.5 (typique latent), on étouffe le bruit pour éviter le clipping au décodage
            dampener = 1.0 - torch.sigmoid((signal_mag - 1.5) * 2.0)
            mask = mask * dampener
            print(f"   🛡️ \033[36m[Anti-Burn]\033[0m Protection Latente Active. Bruit étouffé sur les pics de signal.")

        # Injection
        final_noise = noise * detail_power * frequency_boost * mask
        enhanced_samples = samples + final_noise
        
        XT_Mouchard_Cafard.scan("PHASE 1 (OUTPUT)", enhanced_samples, "Latent après injection sécurisée")
        
        return ({"samples": enhanced_samples},)

# ==============================================================================
# NODE 2: WAN TEMPORAL LOCK PRO (V2 - MONITORING)
# ==============================================================================

class Wan_Temporal_Lock_Pro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "lock_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "motion_threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.2, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stabilize"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def stabilize(self, images, lock_strength, motion_threshold):
        XT_Mouchard_Cafard.scan("PHASE 2 (INPUT)", images, "Images brutes (VAE Output)")
        
        output = torch.zeros_like(images)
        output[0] = images[0]
        
        for i in range(1, len(images)):
            curr = images[i]
            prev = output[i-1]
            
            diff = (curr - prev).abs().mean(dim=-1, keepdim=True)
            
            # Sigmoid Hard : Transition nette entre bruit et mouvement
            lock_mask = 1.0 - torch.sigmoid((diff - motion_threshold) * 50.0)
            
            # Blend
            output[i] = prev * (lock_mask * lock_strength) + curr * (1.0 - (lock_mask * lock_strength))

        XT_Mouchard_Cafard.scan("PHASE 2 (OUTPUT)", output, "Images Stabilisées")
        return (output,)

# ==============================================================================
# NODE 3: WAN OLED DYNAMIX (V2.1 - ARRI ROLLOFF FIX)
# CORRECTION: Compression logarithmique des blancs pour supprimer la saturation.
# ==============================================================================

class Wan_OLED_Dynamix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "oled_black_point": ("FLOAT", {"default": 0.010, "min": 0.0, "max": 0.1, "step": 0.001}),
                "specular_pop": ("FLOAT", {"default": 1.10, "min": 1.0, "max": 1.5, "step": 0.01}),
                "shadow_recovery": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.5, "step": 0.01}),
                "highlight_rolloff": ("FLOAT", {"default": 0.80, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Point de départ de la compression douce des blancs."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_tone_map"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def apply_tone_map(self, images, oled_black_point, specular_pop, shadow_recovery, highlight_rolloff):
        XT_Mouchard_Cafard.scan("PHASE 3 (INPUT)", images, "Entrée ToneMap")
        
        img = images.clone()
        
        # 1. Expansion des Noirs (Plus subtile)
        if oled_black_point > 0:
            scale = 1.0 / (1.0 - oled_black_point)
            img = (img - oled_black_point).clamp(min=0.0) * scale

        # 2. Shadow Recovery (Logique Gamma corrigée)
        if shadow_recovery > 0:
            gamma = 1.0 - (shadow_recovery * 0.3)
            img = torch.pow(img + 1e-6, gamma)

        # 3. SPECULAR POP AVEC ROLLOFF (V2.1)
        # Formule Cinéma : Boost + Shoulder Compression
        
        # Separation Luma
        luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        luma = luma.unsqueeze(-1)
        
        # Boost parabolique (plus fort sur les hautes lumières)
        boost_amount = (specular_pop - 1.0) * luma * luma
        img_boosted = img + boost_amount
        
        # ROLLOFF PROTECTION (Le "Soft Clip")
        # Si la valeur dépasse 'highlight_rolloff' (ex: 0.8), on compresse.
        # Cette formule garantit que l'image ne touchera jamais brutalement 1.0
        
        threshold = highlight_rolloff
        # Masque des pixels qui dépassent le seuil de sécurité
        mask_high = (img_boosted > threshold).float()
        
        # Formule de compression "Reinhard modified" pour la partie haute
        delta = img_boosted - threshold
        # La pente diminue à mesure que l'on monte
        compressed = threshold + (delta / (1.0 + delta * 2.5)) * (1.0 - threshold)
        
        # Fusion : Basse lumière normale + Haute lumière compressée
        img_final = img_boosted * (1.0 - mask_high) + compressed * mask_high
        
        # Sécurité ultime
        img_final = torch.clamp(img_final, 0.0, 1.0)
        
        XT_Mouchard_Cafard.scan("PHASE 3 (OUTPUT)", img_final, "Sortie ToneMap (Rolloff Active)")
        return (img_final,)

# ==============================================================================
# NODE 4: WAN ORGANIC SKIN (V2 - OPTICAL BLEND)
# ==============================================================================

class Wan_Organic_Skin:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "film_iso": ("INT", {"default": 150, "min": 0, "max": 3200, "step": 50}),
                "resolution_aware": ("BOOLEAN", {"default": True}),
                "chroma_grain": ("BOOLEAN", {"default": False}),
                "skin_protection": ("BOOLEAN", {"default": True, "tooltip": "Préserve les tons chair du grain excessif."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_texture"
    CATEGORY = "Wan_Architect/Visual_Supremacy"

    def apply_texture(self, images, film_iso, resolution_aware, chroma_grain, skin_protection):
        XT_Mouchard_Cafard.scan("PHASE 4 (INPUT)", images, "Entrée Texturing")
        
        b, h, w, c = images.shape
        device = images.device
        
        # 1. Scaling Adaptatif
        grain_scale = 1.0
        if resolution_aware:
            pixels = h * w
            if pixels > 1920*1080 * 2.5: grain_scale = 0.5
        
        # 2. Génération Bruit
        intensity = (film_iso / 50000.0) * grain_scale
        if chroma_grain:
            noise = torch.randn((b, h, w, c), device=device)
        else:
            noise = torch.randn((b, h, w, 1), device=device).repeat(1, 1, 1, c)
            
        # 3. LUMINANCE MASKING (Anti-Saturation)
        # Le grain est invisible sur le noir (0.0) et le blanc (1.0)
        luma = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
        luma = luma.unsqueeze(-1)
        
        # Courbe en cloche stricte
        visibility_mask = 1.0 - torch.pow(2.0 * luma - 1.0, 4.0)
        
        # Protection Peau
        if skin_protection:
            r, g, bl = images[..., 0], images[..., 1], images[..., 2]
            skin_map = (r > g) & (g > bl)
            skin_map = skin_map.float().unsqueeze(-1)
            visibility_mask = visibility_mask * (1.0 - (skin_map * 0.3))

        # 4. FUSION OPTIQUE
        grain_layer = noise * intensity * visibility_mask
        
        final = images + grain_layer
        final = torch.clamp(final, 0.0, 1.0)
        
        XT_Mouchard_Cafard.scan("PHASE 4 (OUTPUT)", final, "Rendu Final (Master)")
        
        return (final,)

# ==============================================================================
# MAPPINGS
# ==============================================================================
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