import torch
import comfy.model_management as mm
import torch.nn.functional as F

class Wan_Hardware_Accelerator:
    """
    V6.1: TF32 Global Activation (With State Warning).
    Note: Activating TF32 changes the global PyTorch state for Matrix Multiplications.
    This benefits Ampere+ GPUs significantly but affects all nodes in the process.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_tf32": ("BOOLEAN", {"default": True}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True}),
                "high_precision_norm": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("accelerated_model",)
    FUNCTION = "apply_acceleration"
    CATEGORY = "ComfyWan_Architect/Performance"

    def apply_acceleration(self, model, enable_tf32, cudnn_benchmark, high_precision_norm):
        if enable_tf32 and torch.cuda.is_available():
            # Vérification de l'état précédent pour log
            prev_status = torch.backends.cuda.matmul.allow_tf32
            
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            if not prev_status:
                print(f">> [Wan Accel] \033[93mGLOBAL STATE CHANGE:\033[0m TF32 Enabled for PyTorch Matrix Multiplications.")
        
        if cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        m = model.clone()
        return (m,)

class Wan_Attention_Slicer:
    """
    V6 HPC: SDPA Auto-Detect.
    Utilise Flash Attention (SDPA) par défaut si disponible.
    Ne slice que si explicitement demandé.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "slice_size": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": "0 = Auto (SDPA Fast). 1-4 = Force Low VRAM Mode."}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_attention"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def patch_attention(self, model, slice_size):
        m = model.clone()
        
        # P3 Optimisation: SDPA Detection
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        
        # Si slice_size est 0 (Auto)
        if slice_size == 0:
            if has_sdpa:
                print(f">> [Wan Accel] Using PyTorch 2.0 SDPA (Flash Attention). Fast Mode.")
                # On ne fait rien, ComfyUI utilise SDPA par défaut si on ne touche pas
                # On s'assure juste que memory_efficient_attention est activé dans les options Comfy
                current_options = m.model_options.get("transformer_options", {}).copy()
                # On retire les contraintes de slicing
                if "attention_slice_size" in current_options:
                    del current_options["attention_slice_size"]
                m.model_options["transformer_options"] = current_options
            else:
                # Pas de SDPA ? On met une valeur par défaut sûre
                print(f">> [Wan Accel] Legacy PyTorch detected. Applying default slicing (8).")
                current_options = m.model_options.get("transformer_options", {}).copy()
                current_options["attention_slice_size"] = 8
                m.model_options["transformer_options"] = current_options
        
        # Si slice_size > 0 (Mode Forcé Low VRAM)
        else:
            print(f">> [Wan Accel] Forcing Attention Slice: {slice_size} (Low VRAM).")
            current_options = m.model_options.get("transformer_options", {}).copy()
            current_options["memory_efficient_attention"] = True
            current_options["attention_slice_size"] = slice_size
            m.model_options["transformer_options"] = current_options
            
        return (m,)
