"""
CYBERDYNE SYSTEMS CORP.
MODULE: T-X POLYMETRIC INTERPOLATOR (v2.2)
FUNCTION: DUAL PHASE LATENT BRIDGING + INVERSE STRUCTURAL REPULSION
"""

import torch
import torch.nn.functional as F
import nodes
import node_helpers
import comfy.model_management as mm
import comfy.ldm.wan.vae
import comfy.utils
import sys
import gc
import time

# --- CORE OVERRIDE PROTOCOLS ---
original_encode = comfy.ldm.wan.vae.WanVAE.encode
original_decode = comfy.ldm.wan.vae.WanVAE.decode

def tx_encode_override(self, x):
    self._enc_feat_map = [None] * 64
    self._enc_conv_idx = [0]
    self._enc_conv_num = 64
    t = x.shape[2]
    iter_ = 2 + (t - 2) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        elif i == iter_ - 1:
            out_ = self.encoder(x[:, :, -1:, :, :], feat_cache=[None] * self._enc_conv_num, feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    out_head = out[:, :, :iter_ - 1, :, :]
    out_tail = out[:, :, -1, :, :].unsqueeze(2)
    mu, log_var = torch.cat([self.conv1(out_head), self.conv1(out_tail)], dim=2).chunk(2, dim=1)
    return mu

def tx_decode_override(self, z):
    self._feat_map = [None] * 64
    self._conv_idx = [0]
    self._dec_conv_num = 64
    iter_ = z.shape[2]
    z_head = z[:, :, :-1, :, :]
    z_tail = z[:, :, -1, :, :].unsqueeze(2)
    x = torch.cat([self.conv2(z_head), self.conv2(z_tail)], dim=2)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        elif i == iter_ - 1:
            out_ = self.decoder(x[:, :, -1, :, :].unsqueeze(2), feat_cache=None, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
        else:
            out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    return out

class Wan_TX_Interpolator:
    """
    T-X SERIES: POLYMETRIC INTERPOLATOR
    Capacité : Transition liquide entre deux états via Répulsion Structurelle Inverse.
    Moteur : Injection Native VAE + Boost Dynamique Haute Fréquence.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 81, "min": 5, "max": 4096, "step": 4, "tooltip": "Taille temporelle (doit être 4n+1 pour Wan)."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "start_image": ("IMAGE", ),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Netteté de l'image source."}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.5, "step": 0.05, "tooltip": "1.0=Original, 2.0+=Mouvement High-Speed sans ghosting."}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
                "tile_strategy": (["Auto (Smart VRAM)", "512x512 (Safe)", "1024x1024 (Fast)", "1280x1280 (Ultra)"], {"default": "Auto (Smart VRAM)"}),
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
            self.step_t0 = self.t0
            print(f"\n╔══════════════════════════════════════════════════════════════╗", flush=True)
            print(f"║   [CYBERDYNE T-X] POLYMETRIC INTERPOLATOR (ACTIVE)           ║", flush=True)
            print(f"╚══════════════════════════════════════════════════════════════╝", flush=True)
            return
        current_time = time.time()
        print(f"👉 [T-X] {step_name} | {info} ({current_time - self.step_t0:.2f}s)", flush=True)
        self.step_t0 = current_time

    def _get_dtype(self, precision_str):
        if precision_str == "fp16": return torch.float16
        if precision_str == "bf16": return torch.bfloat16
        return torch.float32

    def _sanitize_tensor(self, tensor, target_w, target_h, device, dtype):
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype)
        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        tensor_bchw = tensor.movedim(-1, 1)
        tensor_resized = F.interpolate(tensor_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return torch.clamp(tensor_resized, 0.0, 1.0).movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        img_bchw = image_tensor.movedim(-1, 1)
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=image_tensor.dtype, device=image_tensor.device).view(1, 1, 3, 3)
        b, c, h, w = img_bchw.shape
        enhanced = F.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        return torch.lerp(img_bchw, enhanced, factor * 0.2).clamp(0,1).movedim(1, -1)

    def execute(self, positive, negative, vae, video_frames, width, height, batch_size, start_image, detail_boost, motion_amp, precision, tile_strategy, end_image=None, clip_vision_start=None, clip_vision_end=None):
        self._log("Init")
        device = mm.get_torch_device()
        target_dtype = self._get_dtype(precision)
        
        # 1. Activation du Patch VAE (Gestion des frames clés)
        comfy.ldm.wan.vae.WanVAE.encode = tx_encode_override
        comfy.ldm.wan.vae.WanVAE.decode = tx_decode_override

        try:
            # 2. Préparation des Frames
            resized_start = self._sanitize_tensor(start_image, width, height, device, target_dtype)
            if detail_boost > 0: resized_start = self._enhance_details(resized_start, detail_boost)

            # Construction du volume d'entrée (Start / Middle Gray / End)
            # Wan utilise un schéma temporel spécifique, on remplit le vide avec du gris neutre (0.5)
            gray_vol = torch.ones((video_frames, height, width, 3), device=device, dtype=target_dtype) * 0.5
            gray_vol[0:1] = resized_start[0:1]
            
            valid_end = end_image is not None
            if valid_end:
                resized_end = self._sanitize_tensor(end_image, width, height, device, target_dtype)
                if detail_boost > 0: resized_end = self._enhance_details(resized_end, detail_boost)
                gray_vol[-1:] = resized_end[-1:]

            self._log("Volume Build", f"{video_frames} frames")

            # 3. Encodage Latent
            if hasattr(mm, "load_models_gpu"): mm.load_models_gpu([vae.patcher] if hasattr(vae, "patcher") else [vae])
            
            # Calcul de la taille de tile
            t_size = 1024 if "1024" in tile_strategy else (1280 if "1280" in tile_strategy else 512)
            
            # Encodage du volume officiel (celui avec le gris)
            official_latent = vae.encode(gray_vol)
            self._log("Encoding", f"Shape: {list(official_latent.shape)}")

            # 4. Algorithme de Répulsion Structurelle Inverse (Motion Boost)
            final_latent = official_latent
            if valid_end and motion_amp > 1.001 and video_frames > 2:
                # Création d'un référentiel linéaire (PPT-style) pour isoler les artefacts de mouvement
                start_l = official_latent[:, :, 0:1]
                end_l = official_latent[:, :, -1:]
                t_steps = torch.linspace(0.0, 1.0, official_latent.shape[2], device=device).view(1, 1, -1, 1, 1)
                linear_latent = start_l * (1 - t_steps) + end_l * t_steps

                # Calcul du vecteur de différence (Anti-Ghosting)
                diff = official_latent - linear_latent
                
                # Séparation de Fréquences (Protection des couleurs, boost des structures)
                h_lat, w_lat = diff.shape[-2], diff.shape[-1]
                low_freq = F.interpolate(diff.view(-1, 16, h_lat, w_lat), size=(h_lat // 8, w_lat // 8), mode='area')
                low_freq = F.interpolate(low_freq, size=(h_lat, w_lat), mode='bilinear').view_as(diff)
                high_freq = diff - low_freq # Contient uniquement les informations de mouvement/structure

                # Application du multiplicateur de force (Scaling agressif)
                boost_scale = (motion_amp - 1.0) * 3.5 
                final_latent = official_latent + (high_freq * boost_scale)
                self._log("Motion Engine", f"ISR Boost: {motion_amp}x")

            # 5. Système de Masquage Temporel (Wan-Specific)
            # On verrouille la frame 0 et la frame finale
            mask = torch.ones((1, video_frames, height // 8, width // 8), device=device)
            mask[:, 0] = 0.0
            if valid_end: mask[:, -1] = 0.0

            # Expansion du masque pour correspondre à la structure latente de Wan (4 frames par bloc)
            m_start = torch.repeat_interleave(mask[:, 0:1], 4, dim=1)
            if valid_end:
                m_end = torch.repeat_interleave(mask[:, -1:], 4, dim=1)
                m_mid = mask[:, 1:-1]
                mask_final = torch.cat([m_start, m_mid, m_end], dim=1)
            else:
                mask_final = torch.cat([m_start, mask[:, 1:]], dim=1)

            # Ajustement final des dimensions du masque
            target_len = final_latent.shape[2] * 4
            if mask_final.shape[1] < target_len:
                pad = torch.ones((1, target_len - mask_final.shape[1], height // 8, width // 8), device=device)
                mask_final = torch.cat([mask_final, pad], dim=1)
            mask_final = mask_final[:, :target_len].view(1, final_latent.shape[2], 4, height // 8, width // 8).transpose(1, 2)

            # 6. Injection Conditioning & Clip Vision
            cv_out = None
            if clip_vision_start is not None:
                cv_out = clip_vision_start
                if clip_vision_end is not None:
                    states = torch.cat([cv_out.penultimate_hidden_states, clip_vision_end.penultimate_hidden_states], dim=-2)
                    cv_out = comfy.clip_vision.Output()
                    cv_out.penultimate_hidden_states = states

            pos_final = node_helpers.conditioning_set_values(positive, {"concat_latent_image": final_latent, "concat_mask": mask_final})
            neg_final = node_helpers.conditioning_set_values(negative, {"concat_latent_image": final_latent, "concat_mask": mask_final})
            
            if cv_out is not None:
                pos_final = node_helpers.conditioning_set_values(pos_final, {"clip_vision_output": cv_out})
                neg_final = node_helpers.conditioning_set_values(neg_final, {"clip_vision_output": cv_out})

            self._log("Complete")
            return (pos_final, neg_final, {"samples": torch.zeros_like(final_latent)})

        finally:
            # Restauration impérative du VAE Original
            comfy.ldm.wan.vae.WanVAE.encode = original_encode
            comfy.ldm.wan.vae.WanVAE.decode = original_decode
            gc.collect()
            mm.soft_empty_cache()

NODE_CLASS_MAPPINGS = {"Wan_TX_Interpolator": Wan_TX_Interpolator}
NODE_DISPLAY_NAME_MAPPINGS = {"Wan_TX_Interpolator": "T-X Polymetric Interpolator (ISR)"}
