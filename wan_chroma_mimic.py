# ==============================================================================
# ARCHITECTURE: WAN CHROMA SENTINEL | WARP CORE ENGINE
# VERSION: 4.0 (LUMA LOCK + GAUSSIAN PYRAMID + VRAM OPTIMIZED)
# ==============================================================================

import torch
import torch.nn.functional as F
import comfy.utils
import sys
import math

class XT_Telemetry:
    """
    Système de télémétrie ultra-léger pour le monitoring en temps réel.
    """
    HEADER = "\033[96m[WAN-SENTINEL]\033[0m"
    
    @staticmethod
    def log(msg, type="INFO"):
        colors = {
            "INFO": "\033[92m",   # Green
            "WARN": "\033[93m",   # Yellow
            "CRIT": "\033[91m",   # Red
            "PROC": "\033[94m"    # Blue
        }
        end = "\033[0m"
        print(f"{XT_Telemetry.HEADER} {colors.get(type, '')}[{type}]{end} {msg}")

class Wan_Chroma_Mimic:
    """
    Wan Chroma Mimic - V4.0 ENGINEERING EDITION.
    Correction colorimétrique vectorielle avec préservation de la luminance (Luma Lock).
    Netteté par séparation de fréquences (High-Pass Gaussien).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                
                "mimic_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Force du transfert de couleur."}),
                "luma_lock": ("BOOLEAN", {"default": True, "tooltip": "CRITIQUE: Si activé, garde l'éclairage original de la vidéo et ne change QUE la couleur. Désactivez pour copier aussi la luminosité (risqué)."}),
                
                "detail_restore": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Restaure le piqué perdu par le débruitage/interpolation."}),
                "contrast_curve": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05, "tooltip": "1.0 = Neutre. >1.0 = Plus de contraste (S-Curve)."}),
                
                "protect_highlights": ("BOOLEAN", {"default": True, "tooltip": "Active le Soft-Clipping pour éviter de brûler les blancs."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("master_video",)
    FUNCTION = "apply_sentinel_process"
    CATEGORY = "Wan_Architect/Skynet"

    # --- COLOR SPACE MATH (SRGB <-> LAB D65) ---
    # Implémentation vectorisée optimisée pour PyTorch
    
    def rgb_to_lab(self, img):
        # RGB Linearization (assuming sRGB input approx)
        mask = img > 0.04045
        img = torch.where(mask, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

        # RGB to XYZ
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        
        # XYZ to LAB
        xyz = torch.stack([x, y, z], dim=-1)
        xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=img.device) # D65 White Point
        xyz = xyz / xyz_ref
        
        mask_xyz = xyz > 0.008856
        xyz = torch.where(mask_xyz, torch.pow(xyz, 1/3), (7.787 * xyz) + (16/116))
        
        l = (116 * xyz[..., 1]) - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        
        return torch.stack([l, a, b], dim=-1)

    def lab_to_rgb(self, lab):
        l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        
        xyz = torch.stack([x, y, z], dim=-1)
        mask_xyz = xyz > 0.206893
        xyz = torch.where(mask_xyz, torch.pow(xyz, 3), (xyz - 16/116) / 7.787)
        
        # D65 White Point scaling
        xyz_ref = torch.tensor([0.95047, 1.00000, 1.08883], device=lab.device)
        xyz = xyz * xyz_ref
        
        # XYZ to RGB
        x_val, y_val, z_val = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        r = 3.2404542 * x_val - 1.5371385 * y_val - 0.4985314 * z_val
        g = -0.9692660 * x_val + 1.8760108 * y_val + 0.0415560 * z_val
        b = 0.0556434 * x_val - 0.2040259 * y_val + 1.0572252 * z_val
        
        rgb = torch.stack([r, g, b], dim=-1)
        
        # RGB Gamma Correction
        mask_rgb = rgb > 0.0031308
        rgb = torch.where(mask_rgb, 1.055 * torch.pow(rgb, 1/2.4) - 0.055, 12.92 * rgb)
        
        return torch.clamp(rgb, 0.0, 1.0)

    # --- Gaussian Kernel Generator (On-the-fly) ---
    def get_gaussian_kernel(self, kernel_size=5, sigma=1.0, channels=3, device="cpu"):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        return gaussian_kernel.to(device)

    def get_moments(self, tensor):
        # Calculer mean et std de manière stable
        mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
        std = torch.std(tensor, dim=(1, 2), keepdim=True)
        return mean, std

    def apply_sentinel_process(self, images, reference_image, mimic_intensity, luma_lock, detail_restore, contrast_curve, protect_highlights):
        
        device = images.device
        batch_size = images.shape[0]
        
        XT_Telemetry.log(f"Initializing Sequence: {batch_size} frames on {device}", "INFO")

        # 1. ANALYSE REFERENCE (Calcul unique)
        ref_clean = reference_image[0:1].to(device) # On prend la 1ere frame de ref
        ref_lab = self.rgb_to_lab(ref_clean)
        ref_mean, ref_std = self.get_moments(ref_lab)
        
        # 2. PRÉPARATION DU NOYAU GAUSSIEN (Pour Detail Restore)
        if detail_restore > 0:
            kernel = self.get_gaussian_kernel(kernel_size=7, sigma=1.5, channels=3, device=device)
        
        processed_chunks = []
        chunk_size = 8 # Batch size de sécurité pour 24GB VRAM
        
        pbar = comfy.utils.ProgressBar(batch_size)
        
        for i in range(0, batch_size, chunk_size):
            # --- LOADING CHUNK ---
            chunk = images[i : i + chunk_size].to(device)
            current_batch_count = chunk.shape[0]
            
            # --- ETAPE A: COLOR MATCHING (LAB) ---
            tgt_lab = self.rgb_to_lab(chunk)
            tgt_mean, tgt_std = self.get_moments(tgt_lab)
            
            # Normalisation et Transfert de stats
            # epsilon 1e-6 pour éviter division par zéro
            normalized = (tgt_lab - tgt_mean) / (tgt_std + 1e-6)
            mimic_lab = normalized * ref_std + ref_mean
            
            # Mixage avec l'original selon l'intensité
            mixed_lab = torch.lerp(tgt_lab, mimic_lab, mimic_intensity)
            
            # --- CRITICAL: LUMA LOCK ENGINE ---
            if luma_lock:
                # On remplace le canal L (Luminance) du résultat par le canal L de l'entrée originale
                # Cela garde la structure d'éclairage de la vidéo, mais change la teinte (a, b)
                original_l = tgt_lab[..., 0] # Canal L original
                new_a = mixed_lab[..., 1]    # Canal A modifié
                new_b = mixed_lab[..., 2]    # Canal B modifié
                final_lab = torch.stack([original_l, new_a, new_b], dim=-1)
            else:
                final_lab = mixed_lab

            # Retour en RGB
            final_rgb = self.lab_to_rgb(final_lab)

            # --- ETAPE B: DETAIL RESTORATION (High-Pass Gaussian) ---
            if detail_restore > 0:
                # Permute pour conv2d (B, H, W, C) -> (B, C, H, W)
                input_permuted = final_rgb.permute(0, 3, 1, 2)
                
                # Appliquer flou gaussien (padding reflect pour éviter les bords noirs)
                padded = F.pad(input_permuted, (3,3,3,3), mode='reflect')
                blurred = F.conv2d(padded, kernel, groups=3)
                
                # High pass = Original - Flou
                high_pass = input_permuted - blurred
                
                # Réinjection des détails
                sharpened = input_permuted + (high_pass * detail_restore)
                final_rgb = sharpened.permute(0, 2, 3, 1) # Retour (B, H, W, C)

            # --- ETAPE C: CONTRAST & DYNAMICS ---
            if contrast_curve != 1.0:
                # S-Curve centrée sur 0.5
                final_rgb = final_rgb - 0.5
                final_rgb = final_rgb * contrast_curve
                final_rgb = final_rgb + 0.5

            if protect_highlights:
                # Soft-Clip algorithmique "Shoulder"
                # Compresse doucement les valeurs > 0.9 au lieu de couper net
                knee = 0.9
                mask_high = (final_rgb > knee).float()
                # Formule de compression douce
                compressed = knee + torch.tanh((final_rgb - knee) * 2.0) * (1.0 - knee)
                final_rgb = final_rgb * (1.0 - mask_high) + compressed * mask_high

            # Clamp final de sécurité
            final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
            
            processed_chunks.append(final_rgb)
            pbar.update(current_batch_count)
            
            # Log console discret
            if i % (chunk_size * 2) == 0:
                 sys.stdout.write(f"\r\033[90m[SENTINEL] Processing Frame {i}/{batch_size}\033[0m")
                 sys.stdout.flush()

        print("")
        
        # Assemblage
        final_video = torch.cat(processed_chunks, dim=0)
        
        XT_Telemetry.log("Process Complete. Signal Integrity Verified.", "INFO")
        return (final_video,)

# ==============================================================================
# MAPPINGS
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "Wan_Chroma_Mimic": Wan_Chroma_Mimic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Chroma_Mimic": "Wan Chroma Sentinel 4.0 (Enhanced)"
}
