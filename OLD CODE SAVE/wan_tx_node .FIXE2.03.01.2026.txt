import torch
import torch.nn.functional as F
import nodes
import node_helpers
import comfy.model_management as mm
import comfy.ldm.wan.vae
import gc
import time

# ==============================================================================
# SECTION 1: PROTOCOLES VAE (Logic v2.2 Strict - ANTI-SEPIA)
# ==============================================================================
original_encode = comfy.ldm.wan.vae.WanVAE.encode
original_decode = comfy.ldm.wan.vae.WanVAE.decode

def tx_encode_override(self, x):
    self._enc_feat_map = [None] * 64
    self._enc_conv_idx = [0]
    self._enc_conv_num = 64
    t = x.shape[2]
    iter_ = 2 + (t - 2) // 4
    out = None
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            # Logique v2.2 pour le cache
            slice_start = 1 + 4 * (i - 1)
            slice_end = 1 + 4 * i
            current_x = x[:, :, slice_start:slice_end, :, :]
            out_ = self.encoder(current_x, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    return mu

def tx_decode_override(self, z):
    self._feat_map = [None] * 64
    self._conv_idx = [0]
    self._dec_conv_num = 64
    iter_ = z.shape[2]
    x = self.conv2(z)
    out = None
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    return out

# ==============================================================================
# SECTION 2: NOEUD PRINCIPAL (v3.7 - REPRISE LOGIQUE MASQUE v2.2)
# ==============================================================================

class Wan_TX_Fusion:
    """
    CYBERDYNE SYSTEMS: T-X FUSION (v3.7 Ultimate)
    HYBRID CORE: Restored v2.2 Masking Logic (Fixes End-Freeze) + VAE Patch.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "video_frames": ("INT", {"default": 81, "min": 5, "max": 4096, "step": 4, "tooltip": "Taille temporelle."}),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "start_image": ("IMAGE", ),
                "detail_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "motion_amp": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 3.0, "step": 0.05}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp32"}),
                # Noms conservés pour compatibilité workflow
                "tile_strategy": (["Default (Patch Active)", "Force Tiled (No Patch - Risky)"], {"default": "Default (Patch Active)"}),
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
            print(f"\n[T-X FUSION] 🚀 Initialisation...", flush=True)
            return
        elapsed = time.time() - self.t0
        print(f"👉 [T-X] {step_name:<15} | {info} ({elapsed:.2f}s)", flush=True)

    def _get_dtype(self, precision_str):
        if precision_str == "fp16": return torch.float16
        if precision_str == "bf16": return torch.bfloat16
        return torch.float32

    def _sanitize_tensor(self, tensor, target_w, target_h, device, dtype):
        if tensor.device != device or tensor.dtype != dtype:
            tensor = tensor.to(device=device, dtype=dtype)
        if tensor.dim() == 3: tensor = tensor.unsqueeze(0)
        tensor_bchw = tensor.movedim(-1, 1)
        if tensor_bchw.shape[2] != target_h or tensor_bchw.shape[3] != target_w:
            tensor_resized = F.interpolate(tensor_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            tensor_resized = tensor_bchw
        return torch.clamp(tensor_resized, 0.0, 1.0).movedim(1, -1)

    def _enhance_details(self, image_tensor, factor=0.5):
        if factor <= 0: return image_tensor
        img_bchw = image_tensor.movedim(-1, 1)
        kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
                              dtype=image_tensor.dtype, device=image_tensor.device).view(1, 1, 3, 3)
        b, c, h, w = img_bchw.shape
        enhanced = F.conv2d(img_bchw.reshape(b*c, 1, h, w), kernel, padding=1).view(b, c, h, w)
        return torch.lerp(img_bchw, enhanced, factor * 0.2).clamp(0,1).movedim(1, -1)

    def execute(self, positive, negative, vae, video_frames, width, height, start_image, detail_boost, motion_amp, precision, tile_strategy, end_image=None, clip_vision_start=None, clip_vision_end=None):
        self._log("Init")
        device = mm.get_torch_device()
        target_dtype = self._get_dtype(precision)
        
        # 1. Activation Patch VAE (Mode v2.2)
        use_patch = "Default" in tile_strategy
        if use_patch:
            comfy.ldm.wan.vae.WanVAE.encode = tx_encode_override
            comfy.ldm.wan.vae.WanVAE.decode = tx_decode_override
            self._log("System", "⚡ VAE Patch ACTIVE (Anti-Sepia)")

        try:
            # 2. Préparation (Méthode v2.2 : Gray Volume)
            resized_start = self._sanitize_tensor(start_image, width, height, device, target_dtype)
            if detail_boost > 0: resized_start = self._enhance_details(resized_start, detail_boost)

            # Utilisation du volume gris comme dans la v2.2
            volume = torch.full((video_frames, height, width, 3), 0.5, device=device, dtype=target_dtype)
            volume[0:1] = resized_start[0:1]
            
            valid_end = False
            if end_image is not None:
                resized_end = self._sanitize_tensor(end_image, width, height, device, target_dtype)
                if detail_boost > 0: resized_end = self._enhance_details(resized_end, detail_boost)
                volume[-1:] = resized_end[-1:] 
                valid_end = True

            self._log("Volume Build", f"{video_frames} frames")

            # 3. Encodage
            if hasattr(mm, "load_models_gpu"):
                mm.load_models_gpu([vae.patcher] if hasattr(vae, "patcher") else [vae])

            if use_patch:
                latent = vae.encode(volume[:, :, :, :3])
            else:
                try:
                    tile_args = {"tile_x": 512, "tile_y": 512}
                    latent = vae.encode_tiled(volume[:, :, :, :3], **tile_args)
                except:
                    latent = vae.encode(volume[:, :, :, :3])

            self._log("Encoding", f"Shape: {list(latent.shape)}")

            # 4. ISR (Motion Engine) - v2.2 Logic
            final_latent = latent
            if valid_end and motion_amp > 1.001 and video_frames > 4:
                start_l = latent[:, :, 0:1]
                end_l = latent[:, :, -1:]
                # FIX CPU/GPU pour le linspace
                t_steps = torch.linspace(0.0, 1.0, latent.shape[2], device=latent.device, dtype=latent.dtype).view(1, 1, -1, 1, 1)
                
                linear_latent = start_l * (1 - t_steps) + end_l * t_steps
                diff = latent - linear_latent
                
                h_lat, w_lat = diff.shape[-2], diff.shape[-1]
                low_freq = F.interpolate(diff.view(-1, 16, h_lat, w_lat), size=(h_lat // 8, w_lat // 8), mode='area')
                low_freq = F.interpolate(low_freq, size=(h_lat, w_lat), mode='bilinear').view_as(diff)
                high_freq = diff - low_freq 

                boost_scale = (motion_amp - 1.0) * 2.0
                final_latent = latent + (high_freq * boost_scale * 1.5) + (low_freq * boost_scale * 0.3)
                self._log("Motion Engine", f"ISR Active | Amp: {motion_amp}x")

            # 5. MASQUAGE 5D AVANCE (LA CORRECTION CRITIQUE v2.2)
            # Au lieu d'un masque simple, on recrée la structure en 4 canaux (sub-frames)
            # pour éviter que les 4 dernières frames ne se figent d'un coup.
            
            # Création du masque en résolution temporelle "Pixel"
            mask = torch.ones((1, video_frames, height // 8, width // 8), device=device, dtype=target_dtype)
            mask[:, 0] = 0.0
            if valid_end:
                mask[:, -1] = 0.0

            # Expansion manuelle pour matcher le Latent (4 frames par bloc)
            # C'est cette étape qui permet la fluidité de fin
            m_start = torch.repeat_interleave(mask[:, 0:1], 4, dim=1)
            
            if valid_end:
                m_end = torch.repeat_interleave(mask[:, -1:], 4, dim=1)
                m_mid = mask[:, 1:-1]
                mask_final = torch.cat([m_start, m_mid, m_end], dim=1)
            else:
                mask_final = torch.cat([m_start, mask[:, 1:]], dim=1)

            # Ajustement aux dimensions Latentes
            target_len = final_latent.shape[2] * 4
            if mask_final.shape[1] < target_len:
                pad = torch.ones((1, target_len - mask_final.shape[1], height // 8, width // 8), device=device, dtype=target_dtype)
                mask_final = torch.cat([mask_final, pad], dim=1)
            
            # Reshape Final [1, 4, T, H, W] - C'est ce format que Wan préfère
            mask_final = mask_final[:, :target_len].view(1, final_latent.shape[2], 4, height // 8, width // 8).transpose(1, 2)
            
            # 6. Packaging
            cv_out = None
            if clip_vision_start is not None:
                cv_out = clip_vision_start
                if clip_vision_end is not None:
                    try:
                        states = torch.cat([cv_out.penultimate_hidden_states, clip_vision_end.penultimate_hidden_states], dim=-2)
                        cv_out = comfy.clip_vision.Output()
                        cv_out.penultimate_hidden_states = states
                    except:
                        pass

            pos_final = node_helpers.conditioning_set_values(positive, {"concat_latent_image": final_latent, "concat_mask": mask_final})
            neg_final = node_helpers.conditioning_set_values(negative, {"concat_latent_image": final_latent, "concat_mask": mask_final})

            if cv_out is not None:
                pos_final = node_helpers.conditioning_set_values(pos_final, {"clip_vision_output": cv_out})
                neg_final = node_helpers.conditioning_set_values(neg_final, {"clip_vision_output": cv_out})

            self._log("Complete", "Ready.")
            return (pos_final, neg_final, {"samples": torch.zeros_like(final_latent)})

        finally:
            if use_patch:
                comfy.ldm.wan.vae.WanVAE.encode = original_encode
                comfy.ldm.wan.vae.WanVAE.decode = original_decode
            mm.soft_empty_cache()
            gc.collect()

# ==============================================================================
# SECTION 3: PIXEL PERFECT
# ==============================================================================

class Wan_TX_PixelPerfect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generated_video": ("IMAGE", ),
                "target_end_image": ("IMAGE", ),
                "blend_duration": ("INT", {"default": 12, "min": 2, "max": 60}),
                "blend_mode": (["Ease-In-Out", "Linear"], {"default": "Ease-In-Out"}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pixel_perfect_video", )
    FUNCTION = "apply_blend"
    CATEGORY = "XT-404/Wan2.2"

    def apply_blend(self, generated_video, target_end_image, blend_duration, blend_mode):
        print(f"\n[T-X PIXEL PERFECT] 🔧 Starting...")
        
        total_frames = generated_video.shape[0]
        if total_frames < blend_duration: blend_duration = total_frames // 2

        target_h, target_w = generated_video.shape[1], generated_video.shape[2]
        target_permuted = target_end_image.movedim(-1, 1)
        if target_permuted.shape[2] != target_h or target_permuted.shape[3] != target_w:
            target_resized = F.interpolate(target_permuted, size=(target_h, target_w), mode="bilinear", align_corners=False)
        else:
            target_resized = target_permuted
        
        final_target = target_resized.movedim(1, -1)
        final_target_frame = final_target[-1].unsqueeze(0) 

        if blend_mode == "Linear":
            alphas = torch.linspace(0.0, 1.0, blend_duration)
        else:
            t = torch.linspace(-3.0, 3.0, blend_duration)
            alphas = torch.sigmoid(t)
            alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min())

        alphas = alphas.to(device=generated_video.device, dtype=generated_video.dtype)
        final_target_frame = final_target_frame.to(device=generated_video.device, dtype=generated_video.dtype)

        output_video = generated_video.clone()
        start_frame_idx = total_frames - blend_duration
        
        for i in range(blend_duration):
            current_frame_idx = start_frame_idx + i
            alpha = alphas[i]
            blended = torch.lerp(output_video[current_frame_idx], final_target_frame[0], alpha.view(1, 1, 1))
            output_video[current_frame_idx] = blended

        output_video[-1] = final_target_frame[0]
        return (output_video, )

NODE_CLASS_MAPPINGS = {
    "Wan_TX_Fusion": Wan_TX_Fusion,
    "Wan_TX_PixelPerfect": Wan_TX_PixelPerfect
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_TX_Fusion": "T-X Fusion (Wan2.1 Ultimate)",
    "Wan_TX_PixelPerfect": "T-X Pixel Perfect (Crossfade)"
}
