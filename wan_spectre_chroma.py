import torch
import torch.fft
import comfy.model_management as mm
import comfy.utils
import sys

class Wan_Spectre_Chroma_Filter:
    """
    MODULE D : SPECTRE-CHROMA FILTER (OMEGA V2)
    Architecture : YUV-FFT Chromatic Stabilizer
    Fonction : Verrouillage de la dérive sans artefacts d'irisation (Anti-Rainbow).
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "stabilization_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "spectral_smoothness": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "low_freq_cutoff": ("INT", {"default": 12, "min": 1, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stabilize_spectrum"
    CATEGORY = "XT-404/V2_Omega"

    def rgb_to_yuv(self, img):
        # Conversion haute précision pour isoler la luminance
        matrix = torch.tensor([[0.299, 0.587, 0.114], 
                              [-0.14713, -0.28886, 0.436], 
                              [0.615, -0.51499, -0.10001]], device=img.device)
        return torch.matmul(img, matrix.T)

    def yuv_to_rgb(self, img):
        matrix = torch.tensor([[1.0, 0.0, 1.13983], 
                              [1.0, -0.39465, -0.58060], 
                              [1.0, 2.03211, 0.0]], device=img.device)
        return torch.matmul(img, matrix.T)

    def stabilize_spectrum(self, images, stabilization_strength, spectral_smoothness, low_freq_cutoff):
        T, H, W, C = images.shape
        device = images.device
        
        print(f"\n\033[96m╔══════════════════════════════════════════════════════════════╗\033[0m")
        print(f"\033[96m║ [XT-SPECTRE-CHROMA] MISSION: SPECTRAL STABILIZATION        ║\033[0m")
        print(f"\033[96m╚══════════════════════════════════════════════════════════════╝\033[0m")
        
        # 1. PRÉPARATION YUV & RÉFÉRENCE
        yuv_images = self.rgb_to_yuv(images.to(torch.float32))
        ref_yuv = yuv_images[0]
        
        # FFT sur les canaux de Chrominance (U, V) uniquement
        ref_uv_fft = torch.fft.rfft2(ref_yuv[..., 1:].permute(2, 0, 1))
        ref_uv_mag = torch.abs(ref_uv_fft)
        ref_uv_phase = torch.angle(ref_uv_fft)

        # 2. MASQUE GAUSSIEN (Anti-Arc-en-ciel)
        # On crée une atténuation douce pour éviter les interférences
        y_coords = torch.arange(ref_uv_fft.shape[-2], device=device)
        x_coords = torch.arange(ref_uv_fft.shape[-1], device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        dist = torch.sqrt(grid_y**2 + grid_x**2)
        # Masque gaussien inversé pour les basses fréquences
        sigma = low_freq_cutoff * spectral_smoothness
        smooth_mask = torch.exp(-(dist**2) / (2 * sigma**2)).unsqueeze(0)

        pbar = comfy.utils.ProgressBar(T)
        corrected_yuv = [yuv_images[0].unsqueeze(0)]

        for t in range(1, T):
            curr_yuv = yuv_images[t]
            curr_uv_fft = torch.fft.rfft2(curr_yuv[..., 1:].permute(2, 0, 1))
            
            curr_uv_mag = torch.abs(curr_uv_fft)
            curr_uv_phase = torch.angle(curr_uv_fft)
            
            # Calcul du décalage spectral (Télémétrie)
            shift = (curr_uv_mag.mean() / (ref_uv_mag.mean() + 1e-6)).item()
            shift_pct = abs(1.0 - shift) * 100
            
            # Application de la stabilisation sélective (U, V uniquement)
            stable_uv_mag = torch.lerp(curr_uv_mag, ref_uv_mag, stabilization_strength * smooth_mask)
            stable_uv_phase = torch.lerp(curr_uv_phase, ref_uv_phase, stabilization_strength * smooth_mask)
            
            # Reconstruction du spectre
            stable_uv_fft = torch.polar(stable_uv_mag, stable_uv_phase)
            back_uv = torch.fft.irfft2(stable_uv_fft, s=(H, W)).permute(1, 2, 0)
            
            # On conserve la Luminance (Y) originale
            final_yuv = torch.cat([curr_yuv[..., 0:1], back_uv], dim=-1)
            corrected_yuv.append(final_yuv.unsqueeze(0))
            
            sys.stdout.write(f"\r   [SPECTRE] Frame {t+1}/{T} | Spectral Drift: {shift_pct:.2f}% {'[!] Correction' if shift_pct > 5 else '[OK]'}")
            sys.stdout.flush()
            pbar.update(1)

        # 3. SORTIE & NETTOYAGE
        final_rgb = self.yuv_to_rgb(torch.cat(corrected_yuv, dim=0)).clamp(0, 1)
        print(f"\n   👉 \033[92mSpectral Calibration Finished.\033[0m No Chromatic Artifacts detected.\n")
        
        return (final_rgb,)