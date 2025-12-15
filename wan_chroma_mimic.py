import torch
import torch.nn.functional as F

class Wan_Chroma_Mimic:
    """
    Wan Chroma Mimic - TURBO OLED EDITION (V4).
    
    Architecture 100% GPU (PyTorch) pour une vitesse temps réel.
    - Filtre Morphologique : Élimine les micro-points noirs/blancs (Artefacts).
    - Surface Blur Intelligent : Lisse la peau et le métal sans flouter les bords.
    - OLED Dynamics : Gestion du contraste pour des noirs profonds et des couleurs vibrantes.
    - Transfert LAB : Copie l'ambiance de la référence sans casser la lumière.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                
                "effect_intensity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, 
                    "tooltip": "Force du mimétisme des couleurs."
                }),
                
                "oled_contrast": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, 
                    "tooltip": "Booste la dynamique (Noirs profonds, Blancs purs). 0.0 = Neutre."
                }),
                
                "skin_metal_smooth": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, 
                    "tooltip": "Lissage des surfaces (enlève le grain). Garde les contours."
                }),
                
                "detail_crispness": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1, 
                    "tooltip": "Réhausse les micro-détails (Piqué Cinéma)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("oled_master_video",)
    FUNCTION = "apply_gpu_mastering"
    CATEGORY = "Wan_Architect/Skynet"

    def rgb_to_lab(self, img):
        # Approximation rapide RGB -> LAB pour GPU
        # (Suffisant pour le color grading et ultra rapide)
        matrix = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ], device=img.device)
        xyz = torch.matmul(img, matrix.T)
        
        # Transformation non-linéaire sRGB
        mask = xyz > 0.008856
        xyz[mask] = torch.pow(xyz[mask], 1/3)
        xyz[~mask] = 7.787 * xyz[~mask] + 16/116
        
        l = 116 * xyz[..., 1] - 16
        a = 500 * (xyz[..., 0] - xyz[..., 1])
        b = 200 * (xyz[..., 1] - xyz[..., 2])
        return torch.stack([l, a, b], dim=-1)

    def lab_to_rgb(self, lab):
        # Approximation rapide LAB -> RGB pour GPU
        l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        
        xyz = torch.stack([x, y, z], dim=-1)
        mask = xyz > 0.206893
        xyz[mask] = torch.pow(xyz[mask], 3)
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787
        
        matrix = torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311]
        ], device=lab.device)
        
        rgb = torch.matmul(xyz, matrix.T)
        return torch.clamp(rgb, 0.0, 1.0)

    def get_moments(self, tensor):
        # Calcul Moyenne et Ecart-Type (Mean/Std)
        # Tensor shape: [Batch, H, W, Channels]
        mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
        std = torch.std(tensor, dim=(1, 2), keepdim=True)
        return mean, std

    def morphological_cleaner(self, img_tensor, strength):
        """
        Supprime les micro-anomalies (points noirs/blancs isolés).
        Simule un filtre Median/Morphologique sur GPU.
        """
        if strength <= 0: return img_tensor
        
        # Format BCHW pour Conv2d
        x = img_tensor.movedim(-1, 1)
        
        # Operation "Opening" (Supprime les points clairs) suivie de "Closing" (Supprime les points noirs)
        # On utilise MaxPool (Dilation) et -MaxPool(-x) (Erosion)
        kernel_size = 3
        pad = 1
        
        # Erosion (Min Filter)
        eroded = -F.max_pool2d(-x, kernel_size, stride=1, padding=pad)
        # Dilation (Max Filter)
        dilated = F.max_pool2d(eroded, kernel_size, stride=1, padding=pad)
        
        # Soft Blend pour ne pas perdre trop de détails
        return torch.lerp(x, dilated, strength * 0.5).movedim(1, -1)

    def smart_surface_blur(self, img_tensor, strength):
        """
        Floute les surfaces plates (peau, murs) mais garde les bords nets.
        """
        if strength <= 0: return img_tensor
        
        x = img_tensor.movedim(-1, 1)
        
        # Création d'un masque de bords (Sobel ou Laplacien simplifié)
        # On utilise la variance locale comme détecteur de bords
        blurred_guide = F.avg_pool2d(x, 3, stride=1, padding=1)
        diff = torch.abs(x - blurred_guide)
        edge_mask = torch.mean(diff, dim=1, keepdim=True)
        
        # Normalisation du masque (0 = plat, 1 = bord)
        edge_mask = torch.clamp(edge_mask * 10.0, 0.0, 1.0)
        
        # Flou plus fort pour le lissage
        smoothed = F.avg_pool2d(x, 5, stride=1, padding=2)
        
        # Mélange : Si c'est un bord, on garde l'original. Si c'est plat, on lisse.
        # Lerp (Lisse, Original, Masque)
        result = smoothed * (1.0 - edge_mask) + x * edge_mask
        
        return torch.lerp(x, result, strength).movedim(1, -1)

    def apply_gpu_mastering(self, images, reference_image, effect_intensity, oled_contrast, skin_metal_smooth, detail_crispness):
        
        device = images.device
        total_frames = len(images)
        print(f"\n>> [CHROMA MIMIC V4] Turbo OLED Mastering on {total_frames} frames (GPU)...")

        # --- 1. ANALYSE REFERENCE (GPU) ---
        # Nettoyage et conversion LAB de la référence
        ref_clean = torch.nan_to_num(reference_image[0:1], nan=0.0).to(device)
        ref_lab = self.rgb_to_lab(ref_clean)
        
        # Extraction des statistiques de couleur (Mean/Std) de la référence
        ref_mean, ref_std = self.get_moments(ref_lab)

        # Buffer de sortie
        mastered_frames = []

        for i in range(total_frames):
            img = images[i:i+1].to(device) # Traitement par batch de 1 pour économiser VRAM
            
            # --- 2. DESPECKLE (Suppression points noirs) ---
            # On applique un nettoyage morphologique léger au début
            # pour éviter que les points noirs ne soient amplifiés par la suite
            img = self.morphological_cleaner(img, strength=0.5)

            # --- 3. COLOR TRANSFER (LAB Reinhard) ---
            # Conversion LAB
            target_lab = self.rgb_to_lab(img)
            tgt_mean, tgt_std = self.get_moments(target_lab)
            
            # Transfert de statistiques (C'est ça qui copie l'ambiance)
            # (X - Mean_src) * (Std_ref / Std_src) + Mean_ref
            # On évite la division par zéro avec un epsilon
            normalized = (target_lab - tgt_mean) / (tgt_std + 1e-6)
            mimic_lab = normalized * ref_std + ref_mean
            
            # Blend avec l'original selon l'intensité
            final_lab = torch.lerp(target_lab, mimic_lab, effect_intensity)
            
            # Retour RGB
            final_rgb = self.lab_to_rgb(final_lab)

            # --- 4. SMART SMOOTHING (Peau/Métal) ---
            # Lissage intelligent qui préserve les contours
            if skin_metal_smooth > 0:
                final_rgb = self.smart_surface_blur(final_rgb, skin_metal_smooth)

            # --- 5. OLED CONTRAST (S-Curve) ---
            if oled_contrast > 0:
                # Courbe en S pour approfondir les noirs et booster les éclats
                # Centré sur 0.5
                final_rgb = final_rgb - 0.5
                # Formule Sigmoid ajustée
                contrast_factor = 1.0 + oled_contrast
                final_rgb = final_rgb * contrast_factor
                final_rgb = final_rgb + 0.5
                # Recalage des points noirs (Black Point Compensation)
                final_rgb = torch.clamp((final_rgb - 0.02) * 1.05, 0.0, 1.0)

            # --- 6. DETAIL CRISPNESS (Unsharp Mask GPU) ---
            if detail_crispness > 0:
                # On extrait les détails par différence de flou
                blurred = F.avg_pool2d(final_rgb.movedim(-1, 1), 3, stride=1, padding=1).movedim(1, -1)
                details = final_rgb - blurred
                # On réinjecte les détails
                final_rgb = final_rgb + (details * detail_crispness)

            # --- 7. FINAL POLISH ---
            # Sécurité et Clamping
            final_rgb = torch.nan_to_num(final_rgb, nan=0.0)
            final_rgb = torch.clamp(final_rgb, 0.0, 1.0)
            
            mastered_frames.append(final_rgb)

        # Assemblage final
        result_tensor = torch.cat(mastered_frames, dim=0)
        
        print(f">> [CHROMA MIMIC] Rendering Complete (GPU).\n")
        return (result_tensor,)
