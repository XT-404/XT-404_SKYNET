import torch
import torch.nn.functional as F

# ==============================================================================
# PROJECT: CYBERDYNE GENISYS [NANO-REPAIR EDITION]
# MODEL: T-3000 (Nanotech Phase Controller)
# STATUS: V2 (TF32 Stability Fix)
# ==============================================================================

class CYBER_HUD:
    """ Interface Tactique Omnisciente (V5) """
    C = {
        "CYAN": "\033[38;5;51m", "RED": "\033[38;5;196m", "GREEN": "\033[38;5;46m",
        "AMBER": "\033[38;5;214m", "PURPLE": "\033[38;5;135m", "GREY": "\033[38;5;240m",
        "WHITE": "\033[38;5;255m", "RESET": "\033[0m", "BOLD": "\033[1m", "DIM": "\033[2m"
    }

    @staticmethod
    def render_dashboard(step, phase, tao, mag, drift, thresh, action, tactic_msg):
        # Calcul visuel des barres (Signal & Load)
        signal_integrity = max(0, (1.0 - (drift * 8))) * 100.0
        sig_blocks = int(signal_integrity / 10)
        sig_viz = "▰" * sig_blocks + "▱" * (10 - sig_blocks)
        
        s_col = CYBER_HUD.C['GREEN']
        if signal_integrity < 75: s_col = CYBER_HUD.C['CYAN']
        if signal_integrity < 40: s_col = CYBER_HUD.C['AMBER']
        if signal_integrity < 15: s_col = CYBER_HUD.C['RED']

        load_pct = min(1.0, (drift / (thresh + 1e-6))) * 100.0
        load_blocks = int(load_pct / 10)
        load_viz = "▮" * load_blocks + " " * (10 - load_blocks)
        l_col = CYBER_HUD.C['GREY']
        if load_pct > 80: l_col = CYBER_HUD.C['RED']
        elif load_pct > 50: l_col = CYBER_HUD.C['AMBER']

        if step == 0:
            tao_s, mag_s, drift_s = "------", "------", "------"
            load_viz, sig_viz = "..........", ".........."
        else:
            tao_s, mag_s, drift_s = f"{tao:.4f}", f"{mag:.4f}", f"{drift:.4f}"

        if phase == "INJECT":
            p_lbl = f"{CYBER_HUD.C['RED']}INJECT{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['RED']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['DIM']
        elif phase == "MOTION":
            p_lbl = f"{CYBER_HUD.C['AMBER']}MOTION{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['AMBER']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['WHITE']
        else:
            p_lbl = f"{CYBER_HUD.C['CYAN']}STABLE{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['CYAN']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['WHITE']

        if "CACHE" in action: act = f"{CYBER_HUD.C['PURPLE']}▓█ CACHED █▓{CYBER_HUD.C['RESET']}"
        elif "LOCK" in action: act = f"{CYBER_HUD.C['RED']}🔒 LOCKED  {CYBER_HUD.C['RESET']}"
        else: act = f"{CYBER_HUD.C['GREY']}●  RECALC  {CYBER_HUD.C['RESET']}"

        SEP = f"{CYBER_HUD.C['GREY']}│{CYBER_HUD.C['RESET']}"
        msg = (f"{CYBER_HUD.C['BOLD']}ST:{step:02d}{CYBER_HUD.C['RESET']} {SEP} {p_lbl} {SEP} {t_msg} {SEP} "
               f"{val_col}T:{tao_s} M:{mag_s}{CYBER_HUD.C['RESET']} {SEP} "
               f"{CYBER_HUD.C['DIM']}Δ:{CYBER_HUD.C['RESET']}{val_col}{drift_s}{CYBER_HUD.C['RESET']}/{thresh:.3f} "
               f"{l_col}[{load_viz}]{CYBER_HUD.C['RESET']} {SEP} SIG:{s_col}{sig_viz}{CYBER_HUD.C['RESET']} {SEP} {act}")
        print(msg)

class GenisysState:
    def __init__(self):
        self.prev_latent = None      
        self.prev_output = None      
        self.accumulated_err = 0.0   
        self.step_counter = 0        
        self.last_timestep = -1.0
        self.momentum = 0 

class Wan_Cyberdyne_Genisys:
    """
    **Wan 2.2 CYBERDYNE GENISYS [NANO-REPAIR]**
    Système de protection actif pour TF32.
    Corrige les artefacts noirs en temps réel en limitant l'amplitude des tenseurs.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "system_status": ("BOOLEAN", {"default": True, "label": "T-3000 ONLINE"}),
                "security_level": ("INT", {"default": 7, "min": 1, "max": 10}),
                "warmup_steps": ("INT", {"default": 6, "min": 0, "max": 50}),
                "kinetic_momentum": ("INT", {"default": 2, "min": 0, "max": 5}),
                "hud_display": ("BOOLEAN", {"default": True, "label": "Enable HUD"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("T3000_Model",)
    FUNCTION = "deploy_genisys"
    CATEGORY = "Wan_Architect/Skynet"

    def deploy_genisys(self, model, system_status, security_level, warmup_steps, kinetic_momentum, hud_display):
        if not system_status:
            return (model,)

        base_threshold = 0.060 - (security_level * 0.005)
        if base_threshold < 0.005: base_threshold = 0.005

        m = model.clone()
        if not hasattr(m, "genisys_core"):
            m.genisys_core = {}

        def t3000_protocol(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})

            # --- [LAYER 1] PROTECTION D'ENTRÉE ---
            # Si des NaNs arrivent, on remet à 0 (Neutre) au lieu de Max (Blanc/Noir saturé)
            if torch.isnan(input_x).any() or torch.isinf(input_x).any():
                if hud_display:
                    print(f"\033[93m[NANO-REPAIR] Input glitch detected. Stabilizing...\033[0m")
                input_x = torch.nan_to_num(input_x, nan=0.0, posinf=0.0, neginf=0.0)
            
            try: ts_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            except: ts_val = 0.0

            try: 
                if "c_crossattn" in c: fid = c["c_crossattn"].data_ptr()
                else: fid = 0
            except: fid = 0
            key = f"UNIT_{fid}"
            
            if key not in m.genisys_core: m.genisys_core[key] = GenisysState()
            state = m.genisys_core[key]

            # Reset sur saut temporel
            if state.last_timestep != -1 and abs(ts_val - state.last_timestep) > 200:
                state.prev_latent = None
                state.accumulated_err = 0.0
                state.step_counter = 0
                state.momentum = 0
            state.last_timestep = ts_val

            # --- A. CALCUL TÉLÉMÉTRIQUE ---
            curr = input_x.detach().float()
            tao = 0.0
            mag = 0.0
            drift = 0.0
            
            # Check dimensions pour éviter crash resize
            if state.prev_latent is not None:
                if state.prev_latent.shape != curr.shape:
                    state.prev_latent = None
                    state.accumulated_err = 0.0
                    state.step_counter = 0
                    state.momentum = 0
            
            if state.prev_latent is not None:
                prev = state.prev_latent
                c_flat = curr.view(-1)
                p_flat = prev.view(-1)
                if c_flat.shape == p_flat.shape:
                    cos = F.cosine_similarity(c_flat.unsqueeze(0), p_flat.unsqueeze(0), eps=1e-6).item()
                    norm_r = torch.norm(c_flat, p=2).item() / (torch.norm(p_flat, p=2).item() + 1e-6)
                    tao = (1.0 - cos) + abs(1.0 - norm_r)
                    diff = (curr - prev).abs().mean()
                    mag = (diff / (curr.abs().mean() + 1e-6)).item()
                    drift = (tao * 0.70) + (mag * 0.30)
                else:
                    state.prev_latent = None
                
            # --- B. MOTEUR D'EXECUTION ---
            def execute_cycle(decision, phase, tactic, force_recalc=False):
                if force_recalc:
                    out = model_function(input_x, timestep, **c)
                    
                    # --- [LAYER 2] PROTECTION DE SORTIE (TF32 FIX) ---
                    # C'est ici que la magie opère.
                    
                    # 1. Détection d'anomalie
                    has_error = torch.isnan(out).any() or torch.isinf(out).any()
                    
                    if has_error:
                        if hud_display:
                            print(f"\033[91m[NANO-REPAIR] TF32 Artifact blocked. Auto-correcting tensor range.\033[0m")
                        
                        # 2. Réparation "Douce" (Soft Fix)
                        # On remplace les erreurs par 0 (gris moyen/neutre) et non pas des valeurs extrêmes.
                        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # 3. Clamping (Le Secret)
                        # On force les valeurs à rester dans une plage "réelle" (-10 à +10).
                        # Les artefacts noirs sont souvent des valeurs à -65000.
                        # En limitant à +/- 10, on force l'image à rester visible.
                        out = torch.clamp(out, min=-10.0, max=10.0)

                    state.prev_output = out
                    state.prev_latent = curr
                    state.accumulated_err = 0.0
                    if decision == "RECALC": state.momentum = kinetic_momentum
                else:
                    state.accumulated_err = drift
                    out = state.prev_output

                if hud_display:
                    CYBER_HUD.render_dashboard(state.step_counter, phase, tao, mag, drift, base_threshold, decision, tactic)
                
                state.step_counter += 1
                return out

            # --- C. ARBRE DE DECISION ---
            if state.prev_latent is None: return execute_cycle("LOCK", "INJECT", "BOOT/RESET", True)
            if state.step_counter < warmup_steps: return execute_cycle("LOCK", "INJECT", "WARMUP SEQ", True)
            if state.momentum > 0:
                state.momentum -= 1
                return execute_cycle("LOCK", "MOTION", f"KINETIC ({state.momentum+1})", True)

            potential_error = state.accumulated_err + drift
            if potential_error < base_threshold: return execute_cycle("CACHE", "STABLE", "OPTIMIZED", False)
            else: return execute_cycle("RECALC", "MOTION", "DRIFT DETECT", True)

        m.set_model_unet_function_wrapper(t3000_protocol)
        return (m,)

NODE_CLASS_MAPPINGS = { "Wan_Cyberdyne_Genisys": Wan_Cyberdyne_Genisys }
NODE_DISPLAY_NAME_MAPPINGS = { "Wan_Cyberdyne_Genisys": "💀 Cyberdyne Genisys [NANO-REPAIR]" }
