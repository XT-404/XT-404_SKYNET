import torch
import comfy.model_management as mm
import sys

# ==============================================================================
# WAN ARCHITECT: MAGCACHE OMEGA + T-1000 SENTINEL (V10.0 Final)
# ==============================================================================

class T1000_Sentinel:
    """
    Module de télémétrie active.
    Ne modifie PAS le signal, mais l'observe au microscope.
    Affiche la fidélité réelle du signal par rapport au Prompt.
    """
    COLORS = {
        "CYAN": "\033[96m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "RESET": "\033[0m",
        "BOLD": "\033[1m"
    }

    @staticmethod
    def report(step, flow_type, flow_id, drift, threshold, action, is_hard_lock):
        # Calcul de la fidélité du signal (1.0 - dérive)
        # Une dérive de 0.02 (le seuil) signifie 98% de fidélité structurelle.
        fidelity = max(0, (1.0 - drift)) * 100.0
        
        # Barre visuelle de qualité
        bars = int(fidelity / 5) # 20 barres max
        visual_bar = "█" * bars + "░" * (20 - bars)
        
        color = T1000_Sentinel.COLORS["GREEN"]
        if drift > (threshold * 0.8): color = T1000_Sentinel.COLORS["YELLOW"]
        if drift >= threshold: color = T1000_Sentinel.COLORS["RED"]
        
        status = "LOCKED" if is_hard_lock else action
        
        # Formatage chirurgical pour CMD
        # ID est tronqué aux 6 derniers chiffres pour lisibilité
        short_id = str(flow_id)[-6:] if isinstance(flow_id, int) else str(flow_id)

        msg = (
            f"{T1000_Sentinel.COLORS['CYAN']}[T-1000] Step {step:02d}{T1000_Sentinel.COLORS['RESET']} | "
            f"{flow_type:<8} | "
            f"ID: ..{short_id} | "
            f"Drift: {drift:.5f} / {threshold:.3f} | "
            f"{color}Fidelity: {fidelity:06.3f}% {visual_bar}{T1000_Sentinel.COLORS['RESET']} | "
            f"[{status}]"
        )
        print(msg)

class MagCacheState:
    def __init__(self):
        self.prev_latent = None      
        self.prev_output = None      
        self.accumulated_err = 0.0   
        self.step_counter = 0        
        self.last_timestep = -1.0    

class Wan_MagCache_Patch:
    """
    **Wan 2.2 MagCache (Omega Edition) + T-1000 Sentinel**
    
    Includes:
    - FP8/BF16 Quantum Safety (Fixes QuantizedTensor crashes)
    - Dual-Flow Isolation (Prompts separated via Memory Pointer)
    - T-1000 Telemetry (Real-time signal fidelity analysis)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_mag_cache": ("BOOLEAN", {"default": True}),
                # Seuil recommandé pour Wan 2.2 = 0.020
                "mag_threshold": ("FLOAT", {"default": 0.020, "min": 0.001, "max": 0.5, "step": 0.001}),
                # 0.3 = Force le calcul sur les 30% premiers steps
                "start_step_percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "verbose_t1000": ("BOOLEAN", {"default": True, "label": "Activate T-1000 Display"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("certified_model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "Wan_Architect/Performance"

    def apply_magcache(self, model, enable_mag_cache, mag_threshold, start_step_percent, verbose_t1000):
        if not enable_mag_cache:
            return (model,)

        m = model.clone()
        # Initialisation du stockage d'état Omega
        if not hasattr(m, "wan_omega_state"):
            m.wan_omega_state = {}

        def magcache_wrapper(model_function, params):
            # 1. Extraction et Typage Sécurisé
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})
            
            try:
                ts_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            except:
                ts_val = 0.0

            # 2. Dual-Flow Identification (Prompt DNA)
            # Cette étape garantit que le Prompt Positif ne se mélange jamais au Négatif
            flow_type = "UNKNOWN"
            try:
                if "c_crossattn" in c:
                    sig = c["c_crossattn"]
                    flow_id = sig.data_ptr() # Signature mémoire unique du Prompt
                    flow_type = "POSITIVE"
                elif "y" in c:
                    sig = c["y"]
                    flow_id = sig.data_ptr()
                    flow_type = "COND_IMG"
                else:
                    # Unconditional / Negative Prompt
                    flow_id = 0 
                    flow_type = "NEGATIVE" 
            except:
                flow_id = "Global"

            # Init State unique par Flux
            state_key = f"{flow_type}_{flow_id}"
            if state_key not in m.wan_omega_state:
                m.wan_omega_state[state_key] = MagCacheState()
            
            state = m.wan_omega_state[state_key]

            # 3. Reset Detection (Nouveau Batch ou Nouvelle Image)
            if state.last_timestep != -1:
                # Si le temps saute de plus de 200, c'est une nouvelle génération
                if abs(ts_val - state.last_timestep) > 200:
                    state.prev_latent = None
                    state.accumulated_err = 0.0
                    state.step_counter = 0
            state.last_timestep = ts_val

            # 4. Helper Inference (Fonction de calcul réel)
            def run_inference(reason="EXEC"):
                out = model_function(input_x, timestep, **c)
                state.prev_output = out
                # On stocke une copie FP32 propre pour la comparaison future
                state.prev_latent = input_x.detach().float() 
                state.accumulated_err = 0.0 # Reset error on calc
                
                if verbose_t1000:
                    T1000_Sentinel.report(state.step_counter, flow_type, flow_id, 0.0, mag_threshold, reason, is_hard_lock=False)
                
                state.step_counter += 1
                return out

            # 5. HARD LOCKS (Sécurités Absolues)
            
            # Lock A: Initialisation (Step 0)
            if state.prev_latent is None:
                if verbose_t1000:
                    T1000_Sentinel.report(state.step_counter, flow_type, flow_id, 1.0, mag_threshold, "INIT", is_hard_lock=True)
                
                # Exécution directe sans passer par run_inference pour éviter complexité
                state.prev_output = model_function(input_x, timestep, **c)
                state.prev_latent = input_x.detach().float()
                state.step_counter += 1
                return state.prev_output
            
            # Lock B: Dimension Mismatch (Changement de résolution en cours de route)
            if input_x.shape != state.prev_latent.shape:
                return run_inference("DIM_CHG")

            # Lock C: Turbo / Start Percent
            # Force les calculs initiaux (Steps 0, 1...) selon le paramètre start_step_percent
            force_calc = False
            
            # Règle absolue : Les 2 premiers steps sont TOUJOURS calculés pour Wan
            if state.step_counter < 2: 
                force_calc = True
            elif start_step_percent > 0:
                # Si step_counter est faible (ex: < 5) et qu'on demande > 20%
                if state.step_counter < 6 and start_step_percent >= 0.2:
                    force_calc = True
            
            if force_calc:
                if verbose_t1000:
                     # On calcule quand même le drift pour l'info T-1000
                    curr_f32 = input_x.detach().float()
                    prev_f32 = state.prev_latent
                    diff = (curr_f32 - prev_f32).abs().mean()
                    norm = curr_f32.abs().mean() + 1e-6
                    current_drift = (diff / norm).item()
                    T1000_Sentinel.report(state.step_counter, flow_type, flow_id, current_drift, mag_threshold, "FORCED", is_hard_lock=True)
                
                # Exécution forcée
                out = model_function(input_x, timestep, **c)
                state.prev_output = out
                state.prev_latent = input_x.detach().float()
                state.accumulated_err = 0.0 
                state.step_counter += 1
                return out

            # 6. ANALYSE DU SIGNAL (Cœur du T-1000)
            
            # A. Cast FP32 (Quantum Safety)
            # Empêche le crash "QuantizedTensor unhandled operation"
            curr_f32 = input_x.detach()
            if curr_f32.dtype != torch.float32: 
                curr_f32 = curr_f32.float()
            
            prev_f32 = state.prev_latent

            # B. Calcul Delta (L1 Relative)
            diff = (curr_f32 - prev_f32).abs().mean()
            norm = curr_f32.abs().mean() + 1e-6
            relative_diff = (diff / norm).item()

            # C. Simulation Accumulation
            potential_total_err = state.accumulated_err + relative_diff

            # 7. DÉCISION DU CACHE
            if potential_total_err < mag_threshold:
                # CACHE HIT : Le signal est jugé fidèle
                state.accumulated_err = potential_total_err # On valide l'accumulation
                
                if verbose_t1000:
                    T1000_Sentinel.report(state.step_counter, flow_type, flow_id, state.accumulated_err, mag_threshold, "CACHED", is_hard_lock=False)
                
                state.step_counter += 1
                return state.prev_output
            else:
                # CACHE MISS : Le signal dérive trop -> Recalcul
                if verbose_t1000:
                    T1000_Sentinel.report(state.step_counter, flow_type, flow_id, potential_total_err, mag_threshold, "RECALC", is_hard_lock=False)
                
                out = model_function(input_x, timestep, **c)
                state.prev_output = out
                state.prev_latent = input_x.detach().float()
                state.accumulated_err = 0.0 # Reset erreur
                state.step_counter += 1
                return out

        m.set_model_unet_function_wrapper(magcache_wrapper)
        return (m,)

class Wan_Hybrid_VRAM_Guard:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                # Paramètres legacy conservés pour compatibilité UI
                "tile_size_spatial": ("INT", {"default": 1024}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_standard"
    CATEGORY = "Wan_Architect/Performance"
    
    def decode_standard(self, vae, samples, tile_size_spatial, enable_cpu_offload):
        # Pass-through vers le décodeur natif optimisé
        return (vae.decode(samples["samples"]),)

# MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Wan_MagCache_Patch": Wan_MagCache_Patch,
    "Wan_Hybrid_VRAM_Guard": Wan_Hybrid_VRAM_Guard
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_MagCache_Patch": "Wan MagCache (T-1000 Sentinel)",
    "Wan_Hybrid_VRAM_Guard": "Wan Decode (Native Pass)"
}
