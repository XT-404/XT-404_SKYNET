import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management as mm
import gc

# --- UTILITAIRES SYSTÈME ---

def create_combo_list():
    samplers = comfy.samplers.KSampler.SAMPLERS
    # Wan fonctionne mieux avec 'simple' (Linear). On restreint pour la stabilité.
    schedulers_map = ["simple"] 
    combos = []
    for sch in schedulers_map:
        for sam in samplers:
            combos.append(f"{sch}/{sam}")
    return combos

SKAYNET_COMBOS = create_combo_list()
SAMPLER_MODES = ["standard", "resample", "randomize"] 

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
            return sch, sam 
        return "simple", combo_string

    def compute_dynamic_shift(self, latent_shape, user_shift_override=0.0):
        """
        Calcul automatique du Shift basé sur la résolution (Wan 2.1 Logic).
        Si user_shift_override > 0, on l'utilise. Sinon, calcul dynamique.
        """
        if user_shift_override > 0:
            return user_shift_override

        # Shape [B, C, H, W]
        h, w = latent_shape[2], latent_shape[3]
        pixels = h * w
        base_pixels = 1024 * 1024 # Référence 1MP
        
        # Formule empirique pour Wan : Augmenter le shift pour les hautes résolutions
        # Base shift ~1.0 ou ~5.0 selon les versions. Ici dynamique.
        base_shift = 1.0
        resolution_scale = (pixels / base_pixels) ** 0.5
        
        dynamic_shift = base_shift * resolution_scale
        # Cap de sécurité (min 1.0, max 15.0)
        return max(1.0, min(dynamic_shift, 15.0))

    def compute_wan_sigmas(self, steps, shift=1.0):
        # Calcul haute précision sur CPU
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
        sigmas = (t * shift) / (1 + (shift - 1) * t)
        return sigmas

    def apply_sliding_shield(self, model, strength):
        """
        OPTIMISATION 1: Sliding Window Attention.
        Remplace l'ancre statique par un contexte glissant.
        """
        # Conteneur d'état pour la closure
        state = {"prev_k": None, "prev_v": None}

        def attention_patch(q, k, v, extra_options):
            # Init au premier pass
            if state["prev_k"] is None:
                state["prev_k"] = k.detach()
                state["prev_v"] = v.detach()
                return q, k, v
            
            # Injection de stabilité (Lerp vers le passé immédiat)
            k_stab = torch.lerp(k, state["prev_k"], strength)
            v_stab = torch.lerp(v, state["prev_v"], strength)
            
            # Mise à jour du contexte (Glissement avec decay)
            decay = 0.5 # Facteur d'oubli
            state["prev_k"] = torch.lerp(state["prev_k"], k.detach(), decay)
            state["prev_v"] = torch.lerp(state["prev_v"], v.detach(), decay)
            
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

        # --- 1. GESTION CHAINE / MASTER ---
        if is_chain:
            total_steps = pass_options.get("total_steps", 20)
            start_step = pass_options.get("next_step", 0)
            if "master_sigmas" in pass_options:
                full_sigmas = pass_options["master_sigmas"].to(device="cpu")
        else:
            self.clean_vram(force_unload=False)

        # --- 2. ACTIVATION SHIELD ---
        if temporal_shield > 0:
            model = self.apply_sliding_shield(model, temporal_shield)

        # --- 3. GENERATION SIGMAS (OPTIMISATION 2: DYNAMIC SHIFT) ---
        sch, target_sampler = self.parse_combo(sampler_name)
        
        if not is_chain and full_sigmas is None:
            if sigmas_input is not None:
                 if hasattr(sigmas_input, "shape"):
                     full_sigmas = sigmas_input.clone().cpu().to(dtype=torch.float32)
                     total_steps = len(full_sigmas) - 1
            
            if full_sigmas is None:
                # Calcul Dynamique du Shift si shift_val == 0 ou None
                actual_shift = self.compute_dynamic_shift(latent_image["samples"].shape, shift_val)
                full_sigmas = self.compute_wan_sigmas(total_steps, actual_shift)

        # --- 4. DECOUPAGE STEPS ---
        if steps_to_run == -1: end_step = len(full_sigmas) - 1
        else: end_step = min(start_step + steps_to_run, len(full_sigmas) - 1)

        if start_step >= end_step:
            return (latent_image, latent_image, pass_options)

        current_sigmas = full_sigmas[start_step : end_step + 1]
        latent_tensor = latent_image["samples"].to(device)
        
        # --- 5. ZERO-POINT INJECTION (WAN FIX) ---
        # Wan nécessite que le latent soit vide (0.0) au départ, le bruit est ajouté par le sampler.
        noise = torch.zeros_like(latent_tensor)
        
        if start_step == 0:
            # Master start: On vide le latent et on prépare le bruit
            noise = comfy.sample.prepare_noise(latent_tensor, seed, None)
            latent_tensor = torch.zeros_like(latent_tensor)
        
        # --- 6. SAMPLING ---
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
# NOEUDS SKYNET V2 (MASTER / CHAIN / REFINER)
# ==============================================================================

class XT404_Skynet_V2_1(Skynet_Core_Hybrid_V2):
    """
    MASTER NODE (OMEGA)
    Démarre la séquence, calcule le Dynamic Shift et les Sigmas.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps": ("INT", {"default": 20, "min": 1}), 
                "steps_to_run": ("INT", {"default": 1, "min": -1}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.05}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # 0.0 = AUTO DYNAMIC SHIFT
                "shift_val": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "0.0 = Auto Dynamic (Recommended)"}),
            },
            "optional": {"sigmas": ("SIGMAS",)}
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, sampler_name, steps, steps_to_run, temporal_shield, cfg, seed, shift_val, sigmas=None, **kwargs):
        return self.generic_sample(
            model, seed, steps, cfg, sampler_name, shift_val, positive, negative, latent_image, 1.0, 
            steps_to_run, "standard", True, 0.0, 
            previous_options=None, sigmas_input=sigmas, is_chain=False, node_id="MASTER", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_2(Skynet_Core_Hybrid_V2):
    """
    CHAIN NODE (OMEGA)
    Récupère le contexte du Master et continue le sampling.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps_to_run": ("INT", {"default": 1, "min": 1}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8}),
                "cfg": ("FLOAT", {"default": 2.5}),
                "previous_options": ("DICT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"

    def process(self, model, positive, negative, latent_image, seed_input, sampler_name, steps_to_run, temporal_shield, cfg, previous_options, **kwargs):
        # Shift est ignoré ici car on utilise les sigmas du Master
        return self.generic_sample(
            model, seed_input, 0, cfg, sampler_name, 0.0, positive, negative, latent_image, 1.0, 
            steps_to_run, "standard", True, 0.0, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="CHAIN", temporal_shield=temporal_shield
        )

class XT404_Skynet_V2_3(Skynet_Core_Hybrid_V2):
    """
    REFINER NODE (OMEGA)
    Termine le sampling ou applique un raffinement final.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
                "seed_input": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "sampler_name": (SKAYNET_COMBOS, {"default": "simple/euler"}),
                "steps_to_run": ("INT", {"default": -1, "min": -1}),
                "temporal_shield": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8}),
                "cfg": ("FLOAT", {"default": 2.5}),
                "previous_options": ("DICT",),
            }
        }
    RETURN_TYPES = ("LATENT", "LATENT", "DICT", "INT")
    RETURN_NAMES = ("output", "denoised", "options", "seed_out")
    FUNCTION = "process"
    CATEGORY = "XT-404/V2_Omega"
    
    def process(self, model, positive, negative, latent_image, seed_input, sampler_name, steps_to_run, temporal_shield, cfg, previous_options, **kwargs):
        return self.generic_sample(
            model, seed_input, 0, cfg, sampler_name, 0.0, positive, negative, latent_image, 1.0, 
            steps_to_run, "standard", True, 0.0, 
            previous_options=previous_options, sigmas_input=None, is_chain=True, node_id="REFINER", temporal_shield=temporal_shield
        )

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
