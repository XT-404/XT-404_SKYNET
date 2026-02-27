import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import gc

# --- UTILITAIRES SYSTÈME ---

def create_combo_list():
    samplers = comfy.samplers.KSampler.SAMPLERS
    schedulers_map = ["simple"] 
    combos = []
    for sch in schedulers_map:
        for sam in samplers:
            combos.append(f"{sch}/{sam}")
    return combos

SKAYNET_COMBOS = create_combo_list()
SAMPLER_MODES = ["standard", "resample", "randomize"] 

# --- MOTEUR CORE HYBRIDE OMEGA V2.5 (ULTIMATE COLOR & DYNAMIC FIX) ---

class Skynet_Core_Hybrid_V2:
    
    @staticmethod
    def clean_vram(force_unload=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if force_unload:
            mm.soft_empty_cache()

    def parse_combo(self, combo_string):
        if "/" in combo_string:
            sch, sam = combo_string.split("/")
            return sch, sam 
        return "simple", combo_string

    def compute_dynamic_shift(self, latent_shape, user_shift_override=0.0):
        if user_shift_override > 0:
            return user_shift_override
        # Wan Video: [B, C, T, H, W] -> H, W sont en index 3 et 4
        if len(latent_shape) == 5:
            h, w = latent_shape[3], latent_shape[4]
        else:
            h, w = latent_shape[2], latent_shape[3]
        
        pixels = h * w
        base_pixels = 1024 * 1024 
        base_shift = 1.0
        resolution_scale = (pixels / base_pixels) ** 0.5
        return max(1.0, min(base_shift * resolution_scale, 15.0))

    def compute_wan_sigmas(self, steps, shift=1.0):
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
        sigmas = (t * shift) / (1 + (shift - 1) * t)
        return sigmas

    def apply_sliding_shield(self, model, strength):
        """
        STABILISATION SOFT MOMENTUM :
        On utilise un mélange temporel léger sans écraser les statistiques.
        Mise à jour : Support dynamique pour architecture WAN (Dict format) 
        et isolation des états temporels par Block de réseau.
        """
        states = {}

        def attention_patch(*args, **kwargs):
            q, k, v, extra_options = None, None, None, None
            is_dict_format = False
            payload = None

            # 1. Analyse dynamique des arguments (Modèle Wan vs ComfyUI Standard)
            if len(args) == 1 and isinstance(args[0], dict):
                is_dict_format = True
                payload = args[0]
                q = payload.get("q")
                k = payload.get("k")
                # Dans Wan/DiT, 'x' est souvent utilisé comme 'v'
                v = payload.get("v", payload.get("x"))
                extra_options = payload.get("transformer_options", {})
            elif len(args) >= 3:
                q, k, v = args[0], args[1], args[2]
                extra_options = args[3] if len(args) > 3 else kwargs.get("extra_options", {})
            elif kwargs and "q" in kwargs and "k" in kwargs:
                q = kwargs.get("q")
                k = kwargs.get("k")
                v = kwargs.get("v", kwargs.get("x"))
                extra_options = kwargs.get("extra_options", {})
            else:
                # Fallback de sécurité si signature inconnue
                return args[0] if len(args) == 1 else args

            # 2. Isolation stricte des états par block (évite les corruptions inter-couches)
            block_id = str(extra_options.get("block", "default"))
            k_shape_str = str(k.shape) if k is not None else "None"
            v_shape_str = str(v.shape) if v is not None else "None"
            layer_key = f"{block_id}_{k_shape_str}_{v_shape_str}"

            if layer_key not in states:
                states[layer_key] = {"prev_k": None, "prev_v": None}
            
            state = states[layer_key]

            # 3. Logique de Stabilisation Temporelle
            if state["prev_k"] is None or strength == 0:
                if k is not None: state["prev_k"] = k.detach()
                if v is not None: state["prev_v"] = v.detach()
                if is_dict_format:
                    return payload
                return q, k, v
            
            # Réduction de la force par 2 pour éviter le "Grey Wash"
            actual_strength = strength * 0.5 
            
            # Lissage K
            if k is not None and state["prev_k"] is not None:
                prev_k = state["prev_k"].to(k.device) # Sécurité inter-devices GPU/CPU
                if k.shape == prev_k.shape:
                    k_stab = torch.lerp(k, prev_k, actual_strength)
                    state["prev_k"] = torch.lerp(prev_k, k.detach(), 0.5)
                else:
                    k_stab = k
                    state["prev_k"] = k.detach()
            else:
                k_stab = k

            # Lissage V
            if v is not None and state["prev_v"] is not None:
                prev_v = state["prev_v"].to(v.device) # Sécurité inter-devices GPU/CPU
                if v.shape == prev_v.shape:
                    v_stab = torch.lerp(v, prev_v, actual_strength)
                    state["prev_v"] = torch.lerp(prev_v, v.detach(), 0.5)
                else:
                    v_stab = v
                    state["prev_v"] = v.detach()
            else:
                v_stab = v
            
            # 4. Restitution du format dynamique correct attendu par ComfyUI
            if is_dict_format:
                new_payload = payload.copy()
                if "k" in new_payload: new_payload["k"] = k_stab
                if "v" in new_payload: new_payload["v"] = v_stab
                elif "x" in new_payload: new_payload["x"] = v_stab
                return new_payload
                
            return q, k_stab, v_stab

        m = model.clone()
        m.set_model_attn1_patch(attention_patch)
        return m

    def generic_sample(self, model, seed, steps, cfg, sampler_name, shift_val,
                       positive, negative, latent_image, denoise, steps_to_run, 
                       sampler_mode, bongmath, eta, 
                       previous_options=None, 
                       sigmas_input=None, is_chain=False, node_id="UNK",
                       temporal_shield=0.0):
        
        device = mm.get_torch_device()
        pass_options = previous_options if previous_options is not None else {}
        
        total_steps = steps
        start_step = 0
        full_sigmas = None

        if is_chain:
            total_steps = pass_options.get("total_steps", 20)
            start_step = pass_options.get("next_step", 0)
            if "master_sigmas" in pass_options:
                full_sigmas = pass_options["master_sigmas"].to(device="cpu")
        else:
            self.clean_vram(force_unload=False)

        # Application du Shield stabilisé
        if temporal_shield > 0:
            model = self.apply_sliding_shield(model, temporal_shield)

        sch, target_sampler = self.parse_combo(sampler_name)
        
        if not is_chain and full_sigmas is None:
            if sigmas_input is not None:
                 if hasattr(sigmas_input, "shape"):
                     full_sigmas = sigmas_input.clone().cpu().to(dtype=torch.float32)
                     total_steps = len(full_sigmas) - 1
            
            if full_sigmas is None:
                actual_shift = self.compute_dynamic_shift(latent_image["samples"].shape, shift_val)
                full_sigmas = self.compute_wan_sigmas(total_steps, actual_shift)

        if steps_to_run == -1: end_step = len(full_sigmas) - 1
        else: end_step = min(start_step + steps_to_run, len(full_sigmas) - 1)

        if start_step >= end_step:
            return (latent_image, latent_image, pass_options, seed)

        current_sigmas = full_sigmas[start_step : end_step + 1]
        latent_tensor = latent_image["samples"].to(device)
        
        # Zero-Point Injection
        noise = torch.zeros_like(latent_tensor)
        if start_step == 0:
            noise = comfy.sample.prepare_noise(latent_tensor, seed, None)
            latent_tensor = torch.zeros_like(latent_tensor)
        
        sampler_obj = comfy.samplers.sampler_object(target_sampler)
        
        try:
            samples = comfy.sample.sample_custom(
                model, noise, cfg, sampler_obj, current_sigmas, 
                positive, negative, latent_tensor, 
                noise_mask=None, callback=None, disable_pbar=False, seed=seed
            )
            # --- COLOR FIX --- 
            # Suppression du Clamp destructeur. On laisse Wan s'exprimer librement.
        finally:
            model.set_model_attn1_patch(None)

        out = latent_image.copy()
        out["samples"] = samples
        pass_options.update({"total_steps": total_steps, "next_step": end_step, "master_sigmas": full_sigmas})
        
        return (out, out, pass_options, seed)

# ==============================================================================
# NOEUDS SKYNET V2
# ==============================================================================

class XT404_Skynet_V2_1(Skynet_Core_Hybrid_V2):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps": ("INT", {"default": 20, "min": 1}), 
                "steps_to_run": ("INT", {"default": 1, "min": -1}),
                "temporal_shield": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.8, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "shift_val": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {"sigmas": ("SIGMAS",)}
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, sampler_name, steps, steps_to_run, temporal_shield, cfg, seed, shift_val, sigmas=None):
        return self.generic_sample(
            model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, 
            shift_val=shift_val, positive=positive, negative=negative, latent_image=latent_image, 
            denoise=1.0, steps_to_run=steps_to_run, sampler_mode="standard", bongmath=True, 
            eta=0.0, sigmas_input=sigmas, is_chain=False, node_id="MASTER", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_2(Skynet_Core_Hybrid_V2):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps_to_run": ("INT", {"default": 1, "min": 1}),
                "temporal_shield": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.8, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.5}),
                "previous_options": ("DICT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, seed_input, sampler_name, steps_to_run, temporal_shield, cfg, previous_options):
        return self.generic_sample(
            model=model, seed=seed_input, steps=0, cfg=cfg, sampler_name=sampler_name, 
            shift_val=0.0, positive=positive, negative=negative, latent_image=latent_image, 
            denoise=1.0, steps_to_run=steps_to_run, sampler_mode="resample", bongmath=True, 
            eta=0.0, previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="CHAIN", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_3(Skynet_Core_Hybrid_V2):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps_to_run": ("INT", {"default": -1, "min": -1}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.5}),
                "previous_options": ("DICT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"
    
    def process(self, model, positive, negative, latent_image, seed_input, sampler_name, steps_to_run, temporal_shield, cfg, previous_options):
        return self.generic_sample(
            model=model, seed=seed_input, steps=0, cfg=cfg, sampler_name=sampler_name, 
            shift_val=0.0, positive=positive, negative=negative, latent_image=latent_image, 
            denoise=1.0, steps_to_run=steps_to_run, sampler_mode="resample", bongmath=True, 
            eta=0.0, previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="REFINER", temporal_shield=temporal_shield
        )

# --- MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "XT404_Skynet_V2_1": XT404_Skynet_V2_1,
    "XT404_Skynet_V2_2": XT404_Skynet_V2_2,
    "XT404_Skynet_V2_3": XT404_Skynet_V2_3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XT404_Skynet_V2_1": "XT-404 Skynet 1 (Master) [Omega]",
    "XT404_Skynet_V2_2": "XT-404 Skynet 2 (Chain) [Omega]",
    "XT404_Skynet_V2_3": "XT-404 Skynet 3 (Refiner) [Omega]"
}