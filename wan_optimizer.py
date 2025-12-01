import torch
import comfy.model_management as mm
from comfy.sd import VAE
import torch.nn.functional as F

# ==============================================================================
# WAN ARCHITECT: MAGCACHE OMEGA (V9.9) - SIGNAL PROCESSING UNIT
# ==============================================================================

class MagCacheState:
    def __init__(self):
        self.prev_latent = None      # Stockage FP32 pour comparaison
        self.prev_output = None      # Sortie du modèle cached
        self.accumulated_err = 0.0   # MagCache: Erreur accumulée
        self.step_counter = 0        # Compteur de pas interne
        self.last_timestep = -1.0    # Détection de changement de batch

class Wan_MagCache_Patch:
    """
    **Wan 2.2 MagCache (Omega Edition)**
    Transforme le TeaCache en MagCache (Magnitude-based Cache) avec support FP8.
    
    CRITICAL FIXES:
    - Fixe le crash 'QuantizedTensor' en castant l'input en FP32 pour le calcul de diff.
    - Utilise l'erreur accumulée (MagCache logic) au lieu de l'instantatée.
    - Supporte nativement le Turbo 6 steps.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_mag_cache": ("BOOLEAN", {"default": True}),
                "mag_threshold": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.5, "step": 0.001, "tooltip": "Threshold for accumulated error (formerly rel_l1)"}),
                "start_step_percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "tooltip": "Force run for the first X% of steps (0.3 = 30%)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("mag_model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "Wan_Architect/Performance"

    def apply_magcache(self, model, enable_mag_cache, mag_threshold, start_step_percent):
        if not enable_mag_cache:
            return (model,)

        # 1. Zero-Overhead Cloning
        m = model.clone()
        
        # 2. State Initialization (Attached to Model Object)
        if not hasattr(m, "wan_omega_state"):
            m.wan_omega_state = {}

        def magcache_wrapper(model_function, params):
            # --- A. Extraction des données ---
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})
            
            # Gestion robuste du timestep (peut être un Tensor, un float ou un int)
            try:
                ts_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            except:
                ts_val = 0.0

            # --- B. Identification du Flux (Dual-Flow) ---
            # On sépare le cache pour le prompt positif et négatif (CFG)
            # On utilise le pointeur mémoire du cross-attention comme ID unique
            try:
                if "c_crossattn" in c:
                    flow_id = f"flow_{c['c_crossattn'].data_ptr()}"
                elif "y" in c:
                    flow_id = f"flow_{c['y'].data_ptr()}"
                else:
                    flow_id = "global_flow"
            except:
                flow_id = "global_flow"

            # Init State si absent
            if flow_id not in m.wan_omega_state:
                m.wan_omega_state[flow_id] = MagCacheState()
            
            state = m.wan_omega_state[flow_id]

            # --- C. Détection de Reset (Nouveau Batch/Image) ---
            # Si le timestep saute brutalement ou revient en arrière (début d'un nouveau sampling)
            # Note: En diffusion, timestep diminue souvent (1000->0), mais parfois c'est sigma (0->N)
            # Heuristique: Si l'écart est > 200 ou si on revient à un état initial logique
            if state.last_timestep != -1:
                delta_t = abs(ts_val - state.last_timestep)
                if delta_t > 200: # Seuil arbitraire pour détecter un nouveau run
                    state.prev_latent = None
                    state.accumulated_err = 0.0
                    state.step_counter = 0

            state.last_timestep = ts_val

            # --- D. Helper d'exécution (Le Cœur du Calcul) ---
            def run_model_forward():
                # Exécution réelle du modèle
                output = model_function(input_x, timestep, **c)
                
                # Mise à jour du Cache
                state.prev_output = output
                
                # IMPORTANT: On stocke en FP32 pour éviter la dérive de précision
                # et on détache du graphe pour économiser la VRAM
                state.prev_latent = input_x.detach().float()
                
                # Reset de l'erreur accumulée après un calcul réel
                state.accumulated_err = 0.0
                state.step_counter += 1
                return output

            # --- E. Logique Hard Lock (Turbo Safe) ---
            # Force le calcul si on n'a pas d'historique (Step 0)
            if state.prev_latent is None:
                return run_model_forward()

            # Vérification des dimensions (Si l'utilisateur change la résolution à la volée)
            if input_x.shape != state.prev_latent.shape:
                return run_model_forward()

            # Hard Lock basé sur le pourcentage (ex: 30% des steps)
            # Pour Wan Turbo 6 steps, step_counter sera 0, 1, 2...
            # Si start_step_percent = 0.3, on veut au moins 2 steps forcés (index 0 et 1)
            # Heuristique simple pour ComfyUI (qui ne donne pas toujours max_steps) :
            # On force les 2 premiers steps minimum quoi qu'il arrive.
            if state.step_counter < 2 and start_step_percent > 0.0:
                 return run_model_forward()

            # --- F. MAGCACHE METRIC (QUANTUM SAFE) ---
            # C'est ici que ça crashait avant.
            # Fix: On cast l'input actuel en FP32 JUSTE pour le calcul de la métrique.
            
            # 1. Conversion Input -> FP32 (Safe)
            current_latent_f32 = input_x.detach()
            if current_latent_f32.dtype not in [torch.float32, torch.float64]:
                current_latent_f32 = current_latent_f32.float()
            
            # 2. Récupération Previous (Déjà FP32)
            prev_latent_f32 = state.prev_latent

            # 3. Calcul de la Magnitude Relative (L1)
            # Formule: |Curr - Prev| / |Curr|
            # Cela nous donne le pourcentage de changement du signal.
            diff_abs = (current_latent_f32 - prev_latent_f32).abs().mean()
            norm_abs = current_latent_f32.abs().mean() + 1e-6 # Eviter division par zero
            
            current_relative_diff = diff_abs / norm_abs

            # 4. Accumulation de l'erreur (C'est ça qui fait le "MagCache" vs "TeaCache")
            state.accumulated_err += current_relative_diff.item()

            # --- G. Décision : Cache ou Calcul ? ---
            if state.accumulated_err < mag_threshold:
                # SKIP STEP: On renvoie la sortie précédente
                # On incrémente juste le compteur de steps
                state.step_counter += 1
                # Pas de mise à jour de prev_latent (on garde la référence originale pour accumuler l'écart)
                return state.prev_output
            else:
                # RUN STEP: L'erreur est trop grande, on recalcule
                return run_model_forward()

        # Injection du wrapper
        m.set_model_unet_function_wrapper(magcache_wrapper)
        return (m,)

class Wan_Hybrid_VRAM_Guard:
    """
    OMEGA V21: PASS-THROUGH (Standard Decode)
    Garde la compatibilité du workflow mais utilise le décodeur natif optimisé.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                "tile_size_spatial": ("INT", {"default": 1024}), # Ignoré
                "enable_cpu_offload": ("BOOLEAN", {"default": True}), # Ignoré
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_standard"
    CATEGORY = "Wan_Architect/Performance"
    
    def decode_standard(self, vae, samples, tile_size_spatial, enable_cpu_offload):
        # Utilisation standard de ComfyUI VAE Decode qui gère maintenant très bien la VRAM
        return (vae.decode(samples["samples"]),)

# MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Wan_MagCache_Patch": Wan_MagCache_Patch,
    "Wan_Hybrid_VRAM_Guard": Wan_Hybrid_VRAM_Guard
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_MagCache_Patch": "Wan MagCache (Omega Quantized)",
    "Wan_Hybrid_VRAM_Guard": "Wan Decode (Native Pass)"
}
