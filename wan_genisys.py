import torch
import torch.nn.functional as F

# ==============================================================================
# PROJECT: CYBERDYNE GENISYS [OMNISCIENT EDITION]
# MODEL: T-3000 (Nanotech Phase Controller)
# STATUS: GOLDEN MASTER (PATCHED V2 - ANTI-CRASH)
# ==============================================================================

class CYBER_HUD:
    """
    Interface Tactique Omnisciente (V5).
    Affiche la télémétrie complète même pendant les phases de verrouillage.
    Design: Military / Cyberpunk.
    """
    # Palette T-800 Vision
    C = {
        "CYAN":   "\033[38;5;51m",  # Data Stream
        "RED":    "\033[38;5;196m", # Critical / Skynet
        "GREEN":  "\033[38;5;46m",  # Stable
        "AMBER":  "\033[38;5;214m", # Kinetic / Warning
        "PURPLE": "\033[38;5;135m", # Cache Hit
        "GREY":   "\033[38;5;240m", # Dimmed / Structure
        "WHITE":  "\033[38;5;255m", # High Light
        "RESET":  "\033[0m",
        "BOLD":   "\033[1m",
        "DIM":    "\033[2m"
    }

    @staticmethod
    def render_dashboard(step, phase, tao, mag, drift, thresh, action, tactic_msg):
        # --- 1. BARRE D'INTEGRITE (SIGNAL HEALTH) ---
        # 100% = Image Parfaite (Drift 0).
        signal_integrity = max(0, (1.0 - (drift * 8))) * 100.0
        sig_blocks = int(signal_integrity / 10)
        # Rendu graphique : ▰ plein, ▱ vide
        sig_viz = "▰" * sig_blocks + "▱" * (10 - sig_blocks)
        
        # Couleur dynamique du signal
        s_col = CYBER_HUD.C['GREEN']
        if signal_integrity < 75: s_col = CYBER_HUD.C['CYAN']
        if signal_integrity < 40: s_col = CYBER_HUD.C['AMBER']
        if signal_integrity < 15: s_col = CYBER_HUD.C['RED']

        # --- 2. BARRE DE PRESSION (LOAD METER) ---
        # Montre la proximité du seuil de rupture.
        load_pct = min(1.0, (drift / (thresh + 1e-6))) * 100.0
        load_blocks = int(load_pct / 10)
        load_viz = "▮" * load_blocks + " " * (10 - load_blocks)
        
        l_col = CYBER_HUD.C['GREY']
        if load_pct > 80: l_col = CYBER_HUD.C['RED']
        elif load_pct > 50: l_col = CYBER_HUD.C['AMBER']

        # --- 3. FORMATAGE DES VALEURS ---
        # Gestion du Step 0 (Pas de référence)
        if step == 0:
            tao_s, mag_s, drift_s = "------", "------", "------"
            load_viz, sig_viz = "..........", ".........."
        else:
            tao_s = f"{tao:.4f}"
            mag_s = f"{mag:.4f}"
            drift_s = f"{drift:.4f}"

        # --- 4. COLORIMÉTRIE CONTEXTUELLE ---
        if phase == "INJECT":
            p_lbl = f"{CYBER_HUD.C['RED']}INJECT{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['RED']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['DIM'] # Valeurs grisées mais visibles
        elif phase == "MOTION":
            p_lbl = f"{CYBER_HUD.C['AMBER']}MOTION{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['AMBER']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['WHITE']
        else: # STABLE
            p_lbl = f"{CYBER_HUD.C['CYAN']}STABLE{CYBER_HUD.C['RESET']}"
            t_msg = f"{CYBER_HUD.C['CYAN']}{tactic_msg:<12}{CYBER_HUD.C['RESET']}"
            val_col = CYBER_HUD.C['WHITE']

        # --- 5. ETIQUETTE ACTION ---
        if "CACHE" in action:
            act = f"{CYBER_HUD.C['PURPLE']}▓█ CACHED █▓{CYBER_HUD.C['RESET']}"
        elif "LOCK" in action:
            act = f"{CYBER_HUD.C['RED']}🔒 LOCKED  {CYBER_HUD.C['RESET']}"
        else: # Recalc
            act = f"{CYBER_HUD.C['GREY']}●  RECALC  {CYBER_HUD.C['RESET']}"

        # Séparateur Vertical
        SEP = f"{CYBER_HUD.C['GREY']}│{CYBER_HUD.C['RESET']}"

        # --- ASSEMBLAGE FINAL ---
        # Ex: ST:01 │ INJECT │ WARMUP SEQ   │ T:0.012 M:0.045 │ Δ:0.021/0.025 [|||.......] │ SIG:▰▰▰▰▰▱▱▱▱▱ │ 🔒 LOCKED
        msg = (
            f"{CYBER_HUD.C['BOLD']}ST:{step:02d}{CYBER_HUD.C['RESET']} {SEP} "
            f"{p_lbl} {SEP} "
            f"{t_msg} {SEP} "
            f"{val_col}T:{tao_s} M:{mag_s}{CYBER_HUD.C['RESET']} {SEP} "
            f"{CYBER_HUD.C['DIM']}Δ:{CYBER_HUD.C['RESET']}{val_col}{drift_s}{CYBER_HUD.C['RESET']}/{thresh:.3f} "
            f"{l_col}[{load_viz}]{CYBER_HUD.C['RESET']} {SEP} "
            f"SIG:{s_col}{sig_viz}{CYBER_HUD.C['RESET']} {SEP} "
            f"{act}"
        )
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
    **Wan 2.2 CYBERDYNE GENISYS [OMNISCIENT]**
    Le système de cache ultime pour Wan 2.2.
    Adhérence Parfaite au Prompt + Accélération Sécurisée.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "system_status": ("BOOLEAN", {"default": True, "label": "T-3000 ONLINE"}),
                
                # REGLAGE 1: SECURITE (Seuil de déclenchement)
                # 10 = Ultra Strict (0.010), 1 = Laxiste (0.055). Recommandé: 7 (0.025)
                "security_level": ("INT", {"default": 7, "min": 1, "max": 10}),
                
                # REGLAGE 2: INJECTION PROTOCOL
                # Nombre de steps au début où le cache est INTERDIT. 
                # Crucial pour imprimer le prompt (ex: Face Split). Recommandé: 6.
                "warmup_steps": ("INT", {"default": 6, "min": 0, "max": 50}),
                
                # REGLAGE 3: KINETIC MOMENTUM
                # Nombre de frames calculées de force après un mouvement détecté.
                # Empêche de figer une animation en cours. Recommandé: 2.
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

        # Calcul automatique du Seuil (Threshold)
        base_threshold = 0.060 - (security_level * 0.005)
        if base_threshold < 0.005: base_threshold = 0.005

        m = model.clone()
        if not hasattr(m, "genisys_core"):
            m.genisys_core = {}

        def t3000_protocol(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})
            
            try: ts_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)
            except: ts_val = 0.0

            # 1. Identification du Flux (Isolation Prompt Pos/Neg/Img)
            try: 
                if "c_crossattn" in c: fid = c["c_crossattn"].data_ptr()
                else: fid = 0
            except: fid = 0
            key = f"UNIT_{fid}"
            
            if key not in m.genisys_core:
                m.genisys_core[key] = GenisysState()
            state = m.genisys_core[key]

            # 2. Reset Scene si grand saut temporel
            if state.last_timestep != -1 and abs(ts_val - state.last_timestep) > 200:
                state.prev_latent = None
                state.accumulated_err = 0.0
                state.step_counter = 0
                state.momentum = 0
            state.last_timestep = ts_val

            # --- A. CALCUL TÉLÉMÉTRIQUE (OMNISCIENT) ---
            # On calcule les métriques AVANT de décider, pour l'affichage.
            curr = input_x.detach().float()
            tao = 0.0
            mag = 0.0
            drift = 0.0
            
            # [CRITICAL PATCH] Protection contre les changements de résolution
            # Si la taille change (ex: batch size ou H/W), on force un reset pour éviter le crash.
            if state.prev_latent is not None:
                if state.prev_latent.shape != curr.shape:
                    state.prev_latent = None
                    state.accumulated_err = 0.0
                    state.step_counter = 0
                    state.momentum = 0
            
            # On ne peut calculer la différence que si on a une frame précédente VALIDE
            if state.prev_latent is not None:
                prev = state.prev_latent
                
                # 1. Analyse Tao (Structurelle Vectorielle)
                c_flat = curr.view(-1)
                p_flat = prev.view(-1)
                
                # Safe compute
                if c_flat.shape == p_flat.shape:
                    cos = F.cosine_similarity(c_flat.unsqueeze(0), p_flat.unsqueeze(0), eps=1e-6).item()
                    norm_r = torch.norm(c_flat, p=2).item() / (torch.norm(p_flat, p=2).item() + 1e-6)
                    tao = (1.0 - cos) + abs(1.0 - norm_r)
                    
                    # 2. Analyse Mag (Magnitude/Luminance)
                    diff = (curr - prev).abs().mean()
                    mag = (diff / (curr.abs().mean() + 1e-6)).item()
                    
                    # 3. Fusion Score (Poids: 70% Structure / 30% Lumière)
                    drift = (tao * 0.70) + (mag * 0.30)
                else:
                    # Double sécurité (ne devrait pas arriver grâce au patch ci-dessus)
                    state.prev_latent = None
                
            # --- B. MOTEUR D'EXECUTION ---

            def execute_cycle(decision, phase, tactic, force_recalc=False):
                # Si c'est un calcul forcé ou un cache miss
                if force_recalc:
                    out = model_function(input_x, timestep, **c)
                    state.prev_output = out
                    state.prev_latent = curr # Mise à jour référence
                    state.accumulated_err = 0.0
                    
                    # Si c'était un mouvement naturel (RECALC), on active le Momentum
                    if decision == "RECALC": 
                        state.momentum = kinetic_momentum
                else:
                    # Cache Hit
                    state.accumulated_err = drift # On conserve l'erreur actuelle
                    out = state.prev_output

                if hud_display:
                    CYBER_HUD.render_dashboard(
                        state.step_counter, 
                        phase, 
                        tao, 
                        mag, 
                        drift, 
                        base_threshold, 
                        decision, 
                        tactic
                    )
                
                state.step_counter += 1
                return out

            # --- C. ARBRE DE DECISION TACTIQUE ---

            # 1. HARD LOCKS (Initialisation)
            if state.prev_latent is None:
                # Si le patch de sécurité a trigger (changement de taille), on passe ici.
                return execute_cycle("LOCK", "INJECT", "BOOT/RESET", True)
            
            # 2. PHASE 1: INJECTION (Warmup Forcé)
            if state.step_counter < warmup_steps:
                # On force le calcul, mais on affiche les métriques calculées en A.
                return execute_cycle("LOCK", "INJECT", "WARMUP SEQ", True)

            # 3. PHASE 2: KINETIC MOMENTUM (Protection du mouvement)
            if state.momentum > 0:
                state.momentum -= 1
                return execute_cycle("LOCK", "MOTION", f"KINETIC ({state.momentum+1})", True)

            # 4. PHASE 3: ANALYSE & DÉCISION
            potential_error = state.accumulated_err + drift

            if potential_error < base_threshold:
                # >>> CACHE HIT <<<
                return execute_cycle("CACHE", "STABLE", "OPTIMIZED", False)
            else:
                # >>> RECALC <<<
                return execute_cycle("RECALC", "MOTION", "DRIFT DETECT", True)

        m.set_model_unet_function_wrapper(t3000_protocol)
        return (m,)

# MAPPINGS COMFYUI
NODE_CLASS_MAPPINGS = { 
    "Wan_Cyberdyne_Genisys": Wan_Cyberdyne_Genisys 
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "Wan_Cyberdyne_Genisys": "💀 Cyberdyne Genisys [OMNISCIENT]" 
}
