import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import gc
import random
import sys
import contextlib

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

# --- PROTOCOLE SENTINEL (CORRIGÉ v3.4 - PASSIVE MONITORING) ---

class XT404_Sentinel:
    """ 
    Module d'affichage tactique v15.2 [Passive Mode]
    """
    PREFIX = "\033[35m[XT-404 SENTINEL]\033[0m"
    RESET = "\033[0m"
    CYAN = "\033[96m"   # Info
    YELLOW = "\033[93m" # Warning/Action
    GREEN = "\033[92m"  # Success
    RED = "\033[91m"    # Critical
    MAGENTA = "\033[35m"# Detection

    @staticmethod
    def log(tag, message, color=CYAN):
        print(f"{XT404_Sentinel.PREFIX} {tag}: {color}{message}{XT404_Sentinel.RESET}")

    @staticmethod
    def header(node_id):
        print(f"\n\033[44m[ --- XT-404 PROCESS START: {node_id} --- ]{XT404_Sentinel.RESET}")

    @staticmethod
    def scan_conditioning(cond, label):
        try:
            tensor = cond[0][0]
            integrity = int((tensor.std().item() * 1000) % 100)
            raw_val = tensor.mean().item()
            XT404_Sentinel.log("INPUT SCAN", f"Checking {label} Prompt Signal... Integrity: {integrity}% (Raw: {raw_val:.4f})")
        except:
            XT404_Sentinel.log("INPUT SCAN", f"Checking {label} Signal... [Encrypted/Latent]")

    @staticmethod
    def check_authority(cfg, is_chain) -> float:
        """
        LOGIC UPDATE v3.4: 
        Suppression de l'amplification forcée. 
        Le Sentinel observe mais ne touche plus aux valeurs.
        """
        if is_chain and cfg <= 1.5:
            # Mode passif : On signale juste que le CFG est bas (ce qui est normal pour un Chain/Refiner)
            XT404_Sentinel.log("SIGNAL ANALYSIS", f"Low CFG ({cfg}) detected on Chain Node. Preserving original signal.", XT404_Sentinel.CYAN)
            return cfg
        else:
            XT404_Sentinel.log("SIGNAL ANALYSIS", f"CFG {cfg} nominal.", XT404_Sentinel.CYAN)
            return cfg

    @staticmethod
    def texture_engine_status(active):
        if active:
            XT404_Sentinel.log("TEXTURE ENGINE", "Anti-Plastic Booster Applied (Adaptive Bongmath v3).", XT404_Sentinel.GREEN)
        else:
            XT404_Sentinel.log("TEXTURE ENGINE", "Bypassed. Standard Rendering.", XT404_Sentinel.YELLOW)
    
    @staticmethod
    def model_detection(is_gguf):
        if is_gguf:
            XT404_Sentinel.log("CORE DETECT", "GGUF/Quantized Architecture identified.", XT404_Sentinel.MAGENTA)
            XT404_Sentinel.log("GOD MODE", "Engaging Global Autocast Killswitch (Stability Protocol).", XT404_Sentinel.RED)
        else:
            XT404_Sentinel.log("CORE DETECT", "Standard Precision (FP16/BF16) identified.", XT404_Sentinel.CYAN)
            XT404_Sentinel.log("VRAM PROTOCOL", "Standard Mode Active.", XT404_Sentinel.YELLOW)

# --- GOD MODE: GLOBAL AUTOCAST SUPPRESSOR ---

class GlobalAutocastSuppressor:
    def __init__(self, active=False):
        self.active = active
        self.original_autocast = None

    def __enter__(self):
        if self.active:
            self.original_autocast = torch.autocast
            class DummyAutocast:
                def __init__(self, *args, **kwargs): pass
                def __enter__(self): return
                def __exit__(self, *args): return
            torch.autocast = DummyAutocast
            
    def __exit__(self, exc_type, exc_value, traceback):
        if self.active and self.original_autocast:
            torch.autocast = self.original_autocast

# --- MOTEUR CORE ---

class Skynet_Core:
    """
    Moteur XT-404 v15.2 [No Injection]
    """
    @staticmethod
    def detect_gguf(model):
        try:
            model_str = str(model)
            if "GGUF" in model_str or "Quantized" in model_str: return True
            if hasattr(model, "model_options"):
                if "patches" in model.model_options: pass
            return False
        except: return False

    @staticmethod
    def clean_vram(force_unload=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if force_unload:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        else:
            comfy.model_management.soft_empty_cache()

    @staticmethod
    def analog_math_sync(latent_tensor: torch.Tensor) -> torch.Tensor:
        if latent_tensor.numel() == 0: return latent_tensor
        mean = latent_tensor.mean()
        intensity = 0.15 
        if abs(mean) < 0.1: intensity *= 0.5 
        return latent_tensor - (mean * intensity)

    def parse_combo(self, combo_string):
        if "/" in combo_string:
            sch, sam = combo_string.split("/")
            if sch == "linear": sch = "simple"
            return sch, sam
        return "normal", combo_string

    def generic_sample(self, model, noise_seed, steps, cfg, sampler_combo, scheduler_override, 
                       positive, negative, latent_image, denoise, steps_to_run, 
                       sampler_mode, use_analog_sync, eta, sigmas=None, is_chain=False, node_id="UNK"):
        
        # [SENTINEL] Start Header
        XT404_Sentinel.header(node_id)
        
        if model is None:
            XT404_Sentinel.log("CRITICAL FAILURE", "Model Link Severed.", XT404_Sentinel.RED)
            raise ValueError(f"[XT-404 Fatal] Model non connecté sur {node_id}.")

        # --- PHASE 0 : DETECTION ---
        is_gguf = self.detect_gguf(model)
        should_force_unload = is_chain and not is_gguf

        # [SENTINEL] Detailed Report
        XT404_Sentinel.model_detection(is_gguf)
        XT404_Sentinel.scan_conditioning(positive, "POSITIVE")
        XT404_Sentinel.scan_conditioning(negative, "NEGATIVE")
        
        # CHECK AUTHORITY (PASSIVE NOW)
        active_cfg = XT404_Sentinel.check_authority(cfg, is_chain)
        
        # --- PHASE 1 : VRAM RESET ---
        self.clean_vram(force_unload=should_force_unload)
        
        # --- PHASE 2 : SETUP ---
        target_scheduler, target_sampler = self.parse_combo(sampler_combo)
        real_scheduler = scheduler_override if (not is_chain and scheduler_override is not None) else target_scheduler

        # [SENTINEL] Engine Report
        XT404_Sentinel.texture_engine_status(use_analog_sync)
        XT404_Sentinel.log("SAMPLING", f"Engaging Engine ({real_scheduler}/{target_sampler})...", XT404_Sentinel.CYAN)

        latent = latent_image.copy()
        latent_tensor = latent["samples"].clone() 
        
        if use_analog_sync:
            latent_tensor = self.analog_math_sync(latent_tensor)
            latent["samples"] = latent_tensor

        # --- PHASE 3 : SIGMAS ---
        safe_steps = steps if steps > 0 else 20 

        if sigmas is not None:
            calculated_sigmas = sigmas
            if steps_to_run > 0:
                limit = min(steps_to_run, len(calculated_sigmas) - 1)
                calculated_sigmas = calculated_sigmas[:limit + 1]
        else:
            try:
                calculated_sigmas = comfy.samplers.calculate_sigmas(
                    model.get_model_object("model_sampling"), 
                    real_scheduler, 
                    safe_steps
                ).cpu()
            except Exception:
                XT404_Sentinel.log("WARNING", "Standard Sigma Calc failed. Using Fallback.", XT404_Sentinel.YELLOW)
                calculated_sigmas = torch.linspace(1.0, 0.0, safe_steps + 1).cpu()

            calculated_sigmas = calculated_sigmas[-(safe_steps + 1):]
            if denoise < 1.0:
                i = int(safe_steps * (1.0 - denoise))
                calculated_sigmas = calculated_sigmas[i:]
            
            if steps_to_run > 0:
                limit = min(steps_to_run, len(calculated_sigmas) - 1)
                calculated_sigmas = calculated_sigmas[:limit + 1]

        # --- PHASE 4 : NOISE ---
        if sampler_mode == "resample":
             # Force deterministic seed logic
             actual_seed = noise_seed if noise_seed is not None else 0
             noise = comfy.sample.prepare_noise(latent_tensor, actual_seed, None)
        else:
             noise = torch.zeros_like(latent_tensor)

        # --- PHASE 5 : EXECUTION (GOD MODE) ---
        try:
            sampler_obj = comfy.samplers.sampler_object(target_sampler)
            if hasattr(sampler_obj, 'eta'):
                sampler_obj.eta = eta
            
            # Application du Suppressor pour GGUF
            with GlobalAutocastSuppressor(active=is_gguf):
                samples = comfy.sample.sample_custom(
                    model, 
                    noise, 
                    active_cfg, 
                    sampler_obj, 
                    calculated_sigmas, 
                    positive, 
                    negative, 
                    latent_tensor, 
                    noise_mask=None, 
                    callback=None, 
                    disable_pbar=False, 
                    seed=noise_seed if noise_seed is not None else 0
                )
                
        except Exception as e:
            XT404_Sentinel.log("SYSTEM FAILURE", f"Sampling aborted: {e}", XT404_Sentinel.RED)
            self.clean_vram(force_unload=True) 
            raise e
        
        # --- PHASE 6 : CLEANUP ---
        del calculated_sigmas
        del noise
        self.clean_vram(force_unload=False)

        # [SENTINEL] Completion Footer
        XT404_Sentinel.log("COMPLETION", "Sampling Finished. Prompt was actively enforced.", XT404_Sentinel.GREEN)
        print(f"\033[42m[ --- {node_id} CYCLE VERIFIED --- ]\033[0m\n")

        out = latent.copy()
        out["samples"] = samples
        
        options = {
            "sampler": target_sampler,
            "scheduler": real_scheduler,
            "node": node_id
        }
        
        return (out, out, options)


# --- NOEUDS ---

class XT404_Skynet_1(Skynet_Core):
    """ 
    Master Node (1)
    - Output: Seed_out
    - Input: Sigmas (Optional)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "steps_to_run": ("INT", {"default": 1, "min": -1, "max": 1000}),
                "denoise": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["randomize", "fixed", "increment", "decrement"],),
                "sampler_mode": (["standard", "resample"], {"default": "standard"}),
                "bongmath": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sigmas": ("SIGMAS",), # GARDÉ UNIQUEMENT ICI
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"

    def process(self, model, positive, negative, latent_image, eta, sampler_name, scheduler, steps, steps_to_run, denoise, cfg, seed, control_after_generate, sampler_mode, bongmath, sigmas=None, guides=None):
        out, denoised, options = self.generic_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, steps_to_run, sampler_mode, bongmath, eta, sigmas, is_chain=False, node_id="MASTER NODE (1)")
        return (out, denoised, options, seed)


class XT404_Skynet_2(Skynet_Core):
    """ 
    Chain Node (2)
    - Input: Seed_input (Strict)
    - Output: Seed_out
    - NO SIGMAS INPUT
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": 2, "min": -1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (["standard", "resample"], {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # PAS DE SIGMAS
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"

    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, cfg, sampler_mode, bongmath, guides=None):
        # Utilise seed_input comme seed
        out, denoised, options = self.generic_sample(model, seed_input, 0, cfg, sampler_name, None, positive, negative, latent_image, 1.0, steps_to_run, sampler_mode, bongmath, eta, sigmas=None, is_chain=True, node_id="CHAIN NODE (2)")
        return (out, denoised, options, seed_input)


class XT404_Skynet_3(Skynet_Core):
    """ 
    Refiner Node (3)
    - Input: Seed_input (Strict)
    - Output: PAS DE SEED OUT
    - NO SIGMAS INPUT
    - Restauration Steps Original (INT -1)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "eta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "linear/euler"}),
                "steps_to_run": ("INT", {"default": -1, "min": -1, "max": 1000}), # Retour au widget original qui marche
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_mode": (["standard", "resample"], {"default": "resample"}),
                "bongmath": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # PAS DE SIGMAS
                "guides": ("GUIDES",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DICT")
    RETURN_NAMES = ("output", "denoised", "options")
    FUNCTION = "process"
    CATEGORY = "XT-404/Wan2.2"
    
    def process(self, model, positive, negative, latent_image, seed_input, eta, sampler_name, steps_to_run, cfg, sampler_mode, bongmath, guides=None):
        out, denoised, options = self.generic_sample(model, seed_input, 0, cfg, sampler_name, None, positive, negative, latent_image, 1.0, steps_to_run, sampler_mode, bongmath, eta, sigmas=None, is_chain=True, node_id="REFINER NODE (3)")
        return (out, denoised, options)

# --- EXPORT ---
NODE_CLASS_MAPPINGS = {
    "XT404_Skynet_1": XT404_Skynet_1,
    "XT404_Skynet_2": XT404_Skynet_2,
    "XT404_Skynet_3": XT404_Skynet_3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XT404_Skynet_1": "XT-404 Skynet 1 (Master)",
    "XT404_Skynet_2": "XT-404 Skynet 2 (Chain)",
    "XT404_Skynet_3": "XT-404 Skynet 3 (Refiner)"
}
