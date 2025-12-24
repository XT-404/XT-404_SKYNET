"""
XT-404 SKYNET SUITE : GLOBAL INITIALIZATION
Architecture: Cyberdyne Systems Model T-800 / Wan 2.2 Integration
"""

import sys
import re
import io
import subprocess
from contextlib import redirect_stdout

# --- HUD COLOR MATRIX (ANSI) ---
C_RED     = "\033[91m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_BLUE    = "\033[34m"
C_CYAN    = "\033[96m"
C_MAGENTA = "\033[35m"
C_WHITE   = "\033[97m"
C_GREY    = "\033[90m"
C_RESET   = "\033[0m"

# --- CALIBRATION DU CADRE (PIXEL PERFECT) ---
TOTAL_WIDTH = 80
INNER_WIDTH = TOTAL_WIDTH - 4
BORDER_COLOR = C_CYAN

# --- REGISTRES GLOBAUX ---
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
SYSTEM_CHECKLIST = {}

# --- MOTEUR GRAPHIQUE ---
def get_clean_len(text):
    """Retourne la longueur du texte sans les codes ANSI."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', text))

def render_line(content, align="left"):
    """Affiche une ligne avec calcul de padding absolu."""
    visible_len = get_clean_len(content)
    padding = INNER_WIDTH - visible_len
    if padding < 0: padding = 0 
    
    if align == "center":
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"{BORDER_COLOR}║{C_RESET} {' '*left_pad}{content}{' '*right_pad} {BORDER_COLOR}║{C_RESET}")
    else:
        print(f"{BORDER_COLOR}║{C_RESET} {content}{' '*padding} {BORDER_COLOR}║{C_RESET}")

def render_sep():
    print(f"{BORDER_COLOR}╠{'═'*(TOTAL_WIDTH-2)}╣{C_RESET}")

def render_top():
    print(f"\n{BORDER_COLOR}╔{'═'*(TOTAL_WIDTH-2)}╗{C_RESET}")

def render_bottom():
    print(f"{BORDER_COLOR}╚{'═'*(TOTAL_WIDTH-2)}╝{C_RESET}")

def t800_log(name, status, extra=""):
    """Formatage standardisé des logs."""
    is_valid = any(x in status for x in ["ONLINE", "ACTIVE", "OPTIMIZED", "DETECTED", "CALIBRATED", "LOCKED"])
    is_missing = "MISSING" in status
    s_col = C_GREEN if is_valid else (C_YELLOW if is_missing else C_RED)
    dots_len = 38 - len(name)
    if dots_len < 2: dots_len = 2
    dots = f"{C_GREY}{'.' * dots_len}{C_RESET}"
    line = f"{name} {dots} [{s_col}{status}{C_RESET}] {extra}"
    render_line(line)

# ==============================================================================
# SEQUENCE DE DEMARRAGE
# ==============================================================================

render_top()
render_line(f"{C_RED}CYBERDYNE SYSTEMS CORP. {C_GREY}|{C_RED} SERIES T-800 - MODEL 101 {C_GREY}|{C_RED} V3.7{C_RESET}", "center")
render_sep()

ascii_art = [
    r"█ █ ▀█▀ █ █ █▀█ █ █   █▀▀ █ █ █ █ █▄ █ █▀▀ ▀█▀",
    r"▄▀▄  █  ▀▀█ █ █ ▀▀█   ▀▀█ █▀▄  █  █ ▀█ █▀▀  █ ",
    r"▀ ▀  ▀    ▀ ▀▀▀   ▀   ▀▀▀ ▀ ▀  ▀  ▀  ▀ ▀▀▀  ▀ "
]

render_line("")
for l in ascii_art:
    render_line(f"{C_RED}{l}{C_RESET}", "center")
render_line("")

render_line(f"{C_BLUE}SYSTEM BOOT SEQUENCE: INITIALIZED{C_RESET}")
render_sep()

# --- PHASE 1: XT-404 ---
try:
    from .XT404_Skynet_Nodes import NODE_CLASS_MAPPINGS as XT, NODE_DISPLAY_NAME_MAPPINGS as XT_N
    NODE_CLASS_MAPPINGS.update(XT)
    NODE_DISPLAY_NAME_MAPPINGS.update(XT_N)
    t800_log("NEURAL NET CORE (XT-404)", "ONLINE", f"{C_MAGENTA}Prompt-Lock: ACTIVE")
    SYSTEM_CHECKLIST["XT-404 Core"] = True
except ImportError:
    t800_log("NEURAL NET CORE", "CRITICAL FAILURE")
    SYSTEM_CHECKLIST["XT-404 Core"] = False

# --- PHASE 2: INFILTRATION (GGUF) ---
try:
    from .cyberdyne_model_hub import CyberdyneModelHub
    NODE_CLASS_MAPPINGS["CyberdyneModelHub"] = CyberdyneModelHub
    NODE_DISPLAY_NAME_MAPPINGS["CyberdyneModelHub"] = "Cyberdyne Model Hub"
    t800_log("INFILTRATION UNIT (GGUF)", "DETECTED")
    SYSTEM_CHECKLIST["Cyberdyne Hub"] = True
except ImportError:
    t800_log("INFILTRATION UNIT", "MISSING DEP")
    SYSTEM_CHECKLIST["Cyberdyne Hub"] = False

# --- PHASE 3: T-3000 CORE (GENISYS) ---
try:
    from .wan_genisys import NODE_CLASS_MAPPINGS as GEN, NODE_DISPLAY_NAME_MAPPINGS as GEN_N
    NODE_CLASS_MAPPINGS.update(GEN)
    NODE_DISPLAY_NAME_MAPPINGS.update(GEN_N)
    t800_log("T-3000 PHASE CONTROLLER", "ONLINE", f"{C_RED}Omniscient HUD: ACTIVE")
    SYSTEM_CHECKLIST["Cyberdyne Genisys (T-3000)"] = True
except ImportError:
    t800_log("T-3000 PHASE CONTROLLER", "OFFLINE")
    SYSTEM_CHECKLIST["Cyberdyne Genisys (T-3000)"] = False

# --- PHASE 4: SENSORS ---
try:
    from .wan_i2v_tools import Wan_Vision_OneShot_Cache, Wan_Resolution_Savant
    from .wan_text_encoder import Wan_Text_OneShot_Cache
    
    NODE_CLASS_MAPPINGS["Wan_Vision_OneShot_Cache"] = Wan_Vision_OneShot_Cache
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Vision_OneShot_Cache"] = "Wan Vision OneShot Cache"
    NODE_CLASS_MAPPINGS["Wan_Resolution_Savant"] = Wan_Resolution_Savant
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Resolution_Savant"] = "Wan Resolution Savant (Resize)"
    NODE_CLASS_MAPPINGS["Wan_Text_OneShot_Cache"] = Wan_Text_OneShot_Cache
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Text_OneShot_Cache"] = "Wan Text OneShot Cache"
    
    t800_log("OPTICAL SENSORS (VISION)", "CALIBRATED")
    SYSTEM_CHECKLIST["Wan Vision Cache"] = True
    SYSTEM_CHECKLIST["Wan Resolution"] = True
    SYSTEM_CHECKLIST["Wan Text Cache"] = True
except ImportError:
    t800_log("OPTICAL SENSORS", "DAMAGED")
    SYSTEM_CHECKLIST["Wan Vision Cache"] = False
    SYSTEM_CHECKLIST["Wan Resolution"] = False
    SYSTEM_CHECKLIST["Wan Text Cache"] = False

# --- PHASE 4.5: AUTOMATION ---
try:
    from .auto_wan_node import AutoWanImageOptimizer
    from .auto_half_node import AutoHalfSizeImage
    
    NODE_CLASS_MAPPINGS["AutoWanImageOptimizer"] = AutoWanImageOptimizer
    NODE_DISPLAY_NAME_MAPPINGS["AutoWanImageOptimizer"] = "Auto Wan 2.2 Optimizer (Safe Resize)"
    
    NODE_CLASS_MAPPINGS["AutoHalfSizeImage"] = AutoHalfSizeImage
    NODE_DISPLAY_NAME_MAPPINGS["AutoHalfSizeImage"] = "Auto Image Half Size (1/2)"
    
    t800_log("AUTOMATION SUBROUTINES", "ONLINE", f"{C_BLUE}Smart-Scale: READY")
    SYSTEM_CHECKLIST["Wan Auto Helper"] = True
    SYSTEM_CHECKLIST["Auto Half Resizer"] = True
except ImportError:
    t800_log("AUTOMATION SUBROUTINES", "FAILURE")
    SYSTEM_CHECKLIST["Wan Auto Helper"] = False
    SYSTEM_CHECKLIST["Auto Half Resizer"] = False

# --- PHASE 5: WEAPONS ---
try:
    from .wan_accelerator import Wan_Hardware_Accelerator, Wan_Attention_Slicer
    from .wan_cleanup import Wan_Cycle_Terminator
    
    NODE_CLASS_MAPPINGS["Wan_Hardware_Accelerator"] = Wan_Hardware_Accelerator
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Hardware_Accelerator"] = "Wan Hardware Accelerator"
    NODE_CLASS_MAPPINGS["Wan_Attention_Slicer"] = Wan_Attention_Slicer
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Attention_Slicer"] = "Wan Attention Slicer (SDPA)"
    NODE_CLASS_MAPPINGS["Wan_Cycle_Terminator"] = Wan_Cycle_Terminator
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Cycle_Terminator"] = "Wan Cycle Terminator (System Purge)"
    
    t800_log("TARGETING SYSTEMS (ACCEL)", "LOCKED", f"{C_RED}SDPA: READY")
    SYSTEM_CHECKLIST["Wan Accelerator"] = True
    SYSTEM_CHECKLIST["Wan Terminator"] = True
except ImportError:
    t800_log("TARGETING SYSTEMS", "OFFLINE")
    SYSTEM_CHECKLIST["Wan Accelerator"] = False
    SYSTEM_CHECKLIST["Wan Terminator"] = False

# --- PHASE 6: MIMETIC RENDERING (GEN) ---
try:
    from .wan_fast import WanImageToVideoFidelity
    from .nodes_wan_ultra import WanImageToVideoUltra

    NODE_CLASS_MAPPINGS["WanImageToVideoFidelity"] = WanImageToVideoFidelity
    NODE_DISPLAY_NAME_MAPPINGS["WanImageToVideoFidelity"] = "Wan Image To Video (Optimized FP32 High Fidelity)"

    NODE_CLASS_MAPPINGS["WanImageToVideoUltra"] = WanImageToVideoUltra
    NODE_DISPLAY_NAME_MAPPINGS["WanImageToVideoUltra"] = "Wan Image To Video (Ultra HD - Fidelity - Dynamics)"

    t800_log("MIMETIC RENDERING (GEN)", "ONLINE", f"{C_CYAN}FP32 Core: ACTIVE")
    SYSTEM_CHECKLIST["Wan Fidelity Gen"] = True
    SYSTEM_CHECKLIST["Wan Ultra Gen"] = True

except ImportError:
    t800_log("MIMETIC RENDERING", "CRITICAL ERROR")
    SYSTEM_CHECKLIST["Wan Fidelity Gen"] = False
    SYSTEM_CHECKLIST["Wan Ultra Gen"] = False

# --- PHASE 6.5: POLYMETRIC ALLOY (T-X) ---
try:
    from .wan_tx_node import Wan_TX_Interpolator
    NODE_CLASS_MAPPINGS["Wan_TX_Interpolator"] = Wan_TX_Interpolator
    NODE_DISPLAY_NAME_MAPPINGS["Wan_TX_Interpolator"] = "Wan T-X Interpolator (Dual-Phase)"
    
    t800_log("POLYMETRIC ALLOY (T-X)", "ONLINE", f"{C_MAGENTA}Start/End Engine: READY")
    SYSTEM_CHECKLIST["T-X Interpolator"] = True
except ImportError:
    t800_log("POLYMETRIC ALLOY (T-X)", "NOT FOUND")
    SYSTEM_CHECKLIST["T-X Interpolator"] = False

# --- PHASE 7: COMPRESSOR ---
try:
    f = io.StringIO()
    with redirect_stdout(f):
        from .wan_compressor import Wan_Video_Compressor
    NODE_CLASS_MAPPINGS["Wan_Video_Compressor"] = Wan_Video_Compressor
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Video_Compressor"] = "Wan Video Compressor (H.265)"
    t800_log("DATA COMPRESSION (H.265)", "ACTIVE")
    SYSTEM_CHECKLIST["Wan Compressor"] = True
except ImportError:
    t800_log("DATA COMPRESSION", "MISSING DEP")
    SYSTEM_CHECKLIST["Wan Compressor"] = False

# --- PHASE 8: CAMOUFLAGE (COLOR) ---
try:
    try:
        import skimage
    except ImportError:
        render_line(f"{C_YELLOW}>> [CAMOUFLAGE] Installing scikit-image...{C_RESET}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])
        render_line(f"{C_GREEN}>> [CAMOUFLAGE] Install Complete.{C_RESET}")

    from .wan_chroma_mimic import Wan_Chroma_Mimic
    NODE_CLASS_MAPPINGS["Wan_Chroma_Mimic"] = Wan_Chroma_Mimic
    NODE_DISPLAY_NAME_MAPPINGS["Wan_Chroma_Mimic"] = "Wan Chroma Mimic (Color Match)"
    
    t800_log("CAMOUFLAGE UNIT (COLOR)", "ONLINE")
    SYSTEM_CHECKLIST["Wan Chroma Mimic"] = True
except Exception as e:
    t800_log("CAMOUFLAGE UNIT", "MALFUNCTION")
    SYSTEM_CHECKLIST["Wan Chroma Mimic"] = False

# ==============================================================================
# RAPPORT FINAL
# ==============================================================================
render_sep()
render_line("DIAGNOSTIC COMPLETE.")
render_line(f"{C_GREEN}>> {len(NODE_CLASS_MAPPINGS)} COMBAT MODULES INITIALIZED.{C_RESET}")

for name, status in SYSTEM_CHECKLIST.items():
    check = f"{C_GREEN}[V]{C_RESET}" if status else f"{C_RED}[X]{C_RESET}"
    line = f" Check : {check} {C_GREY if status else C_RED}{name}{C_RESET}"
    render_line(line)

render_bottom()
print(f"{C_RED}[T-800]{C_RESET} I'll be back... waiting for prompt.\n")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
