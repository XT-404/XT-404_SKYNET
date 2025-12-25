import torch
import torch.nn.functional as F
import comfy.model_management as mm
import comfy.utils
import sys

class Wan_Infiltration_Upscaler:
    """
    MODULE C : INFILTRATION VAE UPSCALER (OMEGA V4)
    Architecture : Spatio-Temporal Super-Resolution Reconstruction
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "temporal_window": ("INT", {"default": 16, "min": 8, "max": 64, "step": 4}),
                "temporal_overlap": ("INT", {"default": 4, "min": 4, "max": 12, "step": 4}),
                "resampling_mode": (["bicubic", "bilinear", "nearest-exact"], {"default": "bicubic"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "infiltrate_upscale"
    CATEGORY = "XT-404/V2_Omega"

    def infiltrate_upscale(self, latent, vae, upscale_by, tile_size, temporal_window, temporal_overlap, resampling_mode):
        samples = latent["samples"]
        device = mm.get_torch_device()
        
        # B, C, T, H, W (Latent Space)
        B, C, T, H, W = samples.shape
        
        # Calcul de la résolution cible (Upscale)
        native_h, native_w = H * 8, W * 8
        target_height = int(native_h * upscale_by)
        target_width = int(native_w * upscale_by)
        
        # On force un modulo 8 pour la compatibilité vidéo
        target_height = (target_height // 8) * 8
        target_width = (target_width // 8) * 8
        
        out_frames = (T - 1) * 4 + 1
        
        print(f"\n\033[35m╔══════════════════════════════════════════════════════════════╗\033[0m")
        print(f"\033[35m║ [XT-INFILTRATION] MISSION: TRUE SUPER-RESOLUTION           ║\033[0m")
        print(f"\033[35m╚══════════════════════════════════════════════════════════════╝\033[0m")
        print(f"   👉 Native: {native_w}x{native_h} -> \033[1mTarget: {target_width}x{target_height}\033[0m (x{upscale_by})")
        print(f"   👉 Processing {out_frames} frames via Spatio-Temporal Infiltration...")

        # Buffer de sortie en RAM système (Plus grand car Upscalé)
        output_buffer = torch.zeros((B, out_frames, target_height, target_width, 3), device="cpu")
        count_buffer = torch.zeros((B, out_frames, target_height, target_width, 1), device="cpu")

        pbar = comfy.utils.ProgressBar((T + (temporal_window - temporal_overlap) - 1) // (temporal_window - temporal_overlap))
        vae.patcher.model.to(device)
        
        t_step = max(1, temporal_window - temporal_overlap)
        chunk_idx = 0
        
        for t in range(0, T, t_step):
            t_end = min(t + temporal_window, T)
            sys.stdout.write(f"\r\033[90m   [INFILTRATION] Upscaling Chunk {chunk_idx+1} | Latent T: {t}->{t_end}\033[0m")
            sys.stdout.flush()

            latent_chunk = samples[:, :, t:t_end, :, :].to(device)
            
            # 1. DÉCODAGE NATIF
            decoded_chunk = vae.decode(latent_chunk) # [B, T, H, W, 3]
            
            # 2. UPSCALING DU CHUNK (Espace Pixel)
            # On permute pour F.interpolate : [B*T, C, H, W]
            B_c, T_c, H_c, W_c, C_c = decoded_chunk.shape
            chunk_reshaped = decoded_chunk.view(-1, H_c, W_c, C_c).permute(0, 3, 1, 2)
            
            # Upscale haute qualité avec antialiasing
            upscaled_chunk = F.interpolate(
                chunk_reshaped, 
                size=(target_height, target_width), 
                mode=resampling_mode, 
                align_corners=False,
                antialias=True if resampling_mode == "bicubic" else False
            )
            
            # Retour au format [B, T, H, W, 3]
            upscaled_chunk = upscaled_chunk.permute(0, 2, 3, 1).view(B_c, T_c, target_height, target_width, 3).cpu()

            # 3. BLENDING TEMPOREL (Feathering)
            out_t_start = t * 4
            out_t_end = out_t_start + T_c
            
            mask_t = torch.ones((T_c, 1, 1, 1))
            fade_len = temporal_overlap * 4
            if t > 0 and T_c > fade_len:
                mask_t[:fade_len] = torch.linspace(0, 1, fade_len).view(-1, 1, 1, 1)
            if t_end < T and T_c > fade_len:
                mask_t[-fade_len:] = torch.linspace(1, 0, fade_len).view(-1, 1, 1, 1)

            output_buffer[:, out_t_start:out_t_end] += (upscaled_chunk * mask_t)
            count_buffer[:, out_t_start:out_t_end] += mask_t
            
            chunk_idx += 1
            pbar.update(1)
            del latent_chunk, decoded_chunk, upscaled_chunk

        # Finalisation
        final_video = output_buffer / count_buffer.clamp(min=1e-5)
        final_video = final_video.view(-1, target_height, target_width, 3)
        
        print(f"\n   👉 \033[92mSuper-Resolution Complete.\033[0m Final Output: {target_width}x{target_height}\n")
        
        return (final_video.nan_to_num(0.0), target_width, target_height)

NODE_CLASS_MAPPINGS = {"Wan_Infiltration_Upscaler": Wan_Infiltration_Upscaler}