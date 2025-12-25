import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import gc

# --- UTILITAIRES SYSTÈME ---

def create_combo_list():
    samplers = comfy.samplers.KSampler.SAMPLERS
    schedulers_map = ["linear", "karras", "exponential", "sgm_uniform", "simple", "normal", "beta", "ddim_uniform"]
    combos = []
    for sch in schedulers_map:
        for sam in samplers:
            combos.append(f"{sch}/{sam}")
    return combos

SKAYNET_COMBOS = create_combo_list()
SAMPLER_MODES = ["standard", "resample", "randomize"] 

# --- PROTOCOLE LOGGING OMEGA ---

class XT404_Sentinel:
    PREFIX = "\033[36m[XT-404 OMEGA-V2]\033[0m"
    RESET = "\033[0m"
    GREY = "\033[90m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[35m"

    @staticmethod
    def log(tag, message, color=CYAN):
        print(f"{XT404_Sentinel.PREFIX} {XT404_Sentinel.GREY}{tag}:{XT404_Sentinel.RESET} {color}{message}{XT404_Sentinel.RESET}")

# --- MOTEUR CORE HYBRIDE OMEGA ---

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
            return "simple", sam 
        return "simple", combo_string

    def compute_wan_sigmas(self, steps, shift=1.0):
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
        sigmas = (t * shift) / (1 + (shift - 1) * t)
        return sigmas

    def apply_temporal_shield(self, model, strength, options_dict):
        """T-1000 Attention Shield : Ancrage structurel polymétrique."""
        # On stocke l'ancre dans le dictionnaire persistant
        shield_cache = options_dict.get("shield_anchor", None)

        def attention_patch(q, k, v, extra_options):
            nonlocal shield_cache
            # PHASE 1 : Capture de l'ancre (Master Frame)
            if shield_cache is None:
                shield_cache = {"k": k.detach().clone(), "v": v.detach().clone()}
                options_dict["shield_anchor"] = shield_cache
                return q, k, v
            
            # PHASE 2 : Injection (Interpolation)
            target_k = shield_cache["k"]
            target_v = shield_cache["v"]
            
            if k.shape == target_k.shape:
                k = torch.lerp(k, target_k, strength)
                v = torch.lerp(v, target_v, strength)
            return q, k, v

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

        # DIAGNOSTIC SHIELD : Log systématique
        if temporal_shield > 0:
            XT404_Sentinel.log("SHIELD", f"STATUS: ACTIVE (Strength: {temporal_shield:.2f}) | Node: {node_id}", XT404_Sentinel.MAGENTA)
            model = self.apply_temporal_shield(model, temporal_shield, pass_options)
        else:
            XT404_Sentinel.log("SHIELD", f"STATUS: INACTIVE (Value 0.0) | Node: {node_id}", XT404_Sentinel.GREY)

        _, target_sampler = self.parse_combo(sampler_name)
        
        if not is_chain and full_sigmas is None:
            if sigmas_input is not None:
                 if hasattr(sigmas_input, "shape"):
                     full_sigmas = sigmas_input.clone().cpu().to(dtype=torch.float32)
                     total_steps = len(full_sigmas) - 1
            
            if full_sigmas is None:
                full_sigmas = self.compute_wan_sigmas(total_steps, shift_val)

        if steps_to_run == -1: end_step = len(full_sigmas) - 1
        else: end_step = min(start_step + steps_to_run, len(full_sigmas) - 1)

        if start_step >= end_step:
            return (latent_image, latent_image, pass_options)

        current_sigmas = full_sigmas[start_step : end_step + 1]
        latent_tensor = latent_image["samples"].to(device)
        noise = torch.zeros_like(latent_tensor)
        
        if start_step == 0:
            noise = comfy.sample.prepare_noise(latent_tensor, seed, None)
            latent_tensor = torch.zeros_like(latent_tensor)
            XT404_Sentinel.log("INIT", f"Zero-Point Injection ({node_id})", XT404_Sentinel.CYAN)

        sampler_obj = comfy.samplers.sampler_object(target_sampler)
        try:
            samples = comfy.sample.sample_custom(
                model, noise, cfg, sampler_obj, current_sigmas, 
                positive, negative, latent_tensor, 
                noise_mask=None, callback=None, disable_pbar=False, seed=seed
            )
        finally:
            model.set_model_attn1_patch(None)

        out = latent_image.copy()
        out["samples"] = samples
        pass_options.update({"total_steps": total_steps, "next_step": end_step, "master_sigmas": full_sigmas})
        return (out, out, pass_options)

# ==============================================================================
# NOEUDS SKYNET V2
# ==============================================================================

class XT404_Skynet_V2_1:
    def __init__(self): self.engine = Skynet_Core_Hybrid_V2()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}), 
                "steps_to_run": ("INT", {"default": 1, "min": -1, "max": 1000}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_mode": (SAMPLER_MODES, {"default": "standard"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {"sigmas": ("SIGMAS",), "guides": ("GUIDES",)}
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, eta, sampler_name, scheduler, steps, steps_to_run, temporal_shield, denoise, cfg, seed, sampler_mode, bongmath, shift_val, sigmas=None, **kwargs):
        return self.engine.generic_sample(
            model, seed, steps, cfg, sampler_name, shift_val, positive, negative, latent_image, denoise, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=None, sigmas_input=sigmas, is_chain=False, node_id="MASTER", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_2:
    def __init__(self): self.engine = Skynet_Core_Hybrid_V2()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (SAMPLER_MODES, {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {"previous_options": ("DICT",), "guides": ("GUIDES",)}
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, temporal_shield, cfg, sampler_mode, bongmath, shift_val, previous_options=None, **kwargs):
        return self.engine.generic_sample(
            model, seed_input, 0, cfg, sampler_name, shift_val, positive, negative, latent_image, 1.0, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="CHAIN", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_3:
    def __init__(self): self.engine = Skynet_Core_Hybrid_V2()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (SAMPLER_MODES, {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
                "shift_val": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {"previous_options": ("DICT",), "guides": ("GUIDES",)}
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"
    
    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, temporal_shield, cfg, sampler_mode, bongmath, shift_val, previous_options=None, **kwargs):
        return self.engine.generic_sample(
            model, seed_input, 0, cfg, sampler_name, shift_val, positive, negative, latent_image, 1.0, 
            steps_to_run, sampler_mode, bongmath, eta, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="REFINER", temporal_shield=temporal_shield
        )

NODE_CLASS_MAPPINGS = {
    "XT404_Skynet_V2_1": XT404_Skynet_V2_1,
    "XT404_Skynet_V2_2": XT404_Skynet_V2_2,
    "XT404_Skynet_V2_3": XT404_Skynet_V2_3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XT404_Skynet_V2_1": "XT-404 Skynet 1 (Master) [V2-Shield]",
    "XT404_Skynet_V2_2": "XT-404 Skynet 2 (Chain) [V2-Shield]",
    "XT404_Skynet_V2_3": "XT-404 Skynet 3 (Refiner) [V2-Shield]"
}