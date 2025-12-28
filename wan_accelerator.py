import torch
import torch.nn as nn
import torch.nn.functional as F

class Wan_Hardware_Accelerator:
    """
    OMEGA EDITION V5: Local Precision Casting.
    Solution 'Anti-Burn' sans latence CPU.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_fast_fp32": ("BOOLEAN", {"default": True, "tooltip": "Active le TF32 global, mais protège les LayerNorms en FP32 natif localement."}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("accelerated_model",)
    FUNCTION = "apply_acceleration"
    CATEGORY = "ComfyWan_Architect/Performance"

    def _stabilize_precision_local(self, module):
        """
        Remplace la méthode forward pour forcer le FP32 localement sans toucher aux flags globaux.
        """
        for name, child in module.named_children():
            if isinstance(child, (nn.GroupNorm, nn.LayerNorm)):
                if not hasattr(child, "original_forward"):
                    child.original_forward = child.forward

                # WRAPPER OPTIMISÉ (Zéro overhead backend)
                def secure_forward(x, *args, _layer=child, **kwargs):
                    # On sauvegarde le type d'entrée (ex: FP16 ou BF16)
                    input_dtype = x.dtype
                    
                    # 1. Upcast en FP32 (Léger coût mémoire, gain qualité/stabilité massif)
                    x_32 = x.float()
                    
                    # 2. Exécution de la couche en précision native (Safe)
                    res = _layer.original_forward(x_32, *args, **kwargs)
                    
                    # 3. Downcast vers le type original pour la suite du réseau
                    return res.to(dtype=input_dtype)
                
                child.forward = secure_forward
            else:
                self._stabilize_precision_local(child)

    def apply_acceleration(self, model, enable_fast_fp32, cudnn_benchmark):
        m = model.clone()

        if torch.cuda.is_available():
            # Activation globale pour les convolutions (Vitesse)
            if enable_fast_fp32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            if cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

            # Application du patch chirurgical sur les Norms
            if hasattr(m.model, "diffusion_model"):
                self._stabilize_precision_local(m.model.diffusion_model)
                print(f"\033[92m[ACCEL PRIME]\033[0m Local FP32 Casting injected in Norm Layers.")

        return (m,)

class Wan_Attention_Slicer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "slice_size": ("INT", {"default": 0, "min": 0, "max": 32, "tooltip": "0 = Auto."}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_attention"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def patch_attention(self, model, slice_size):
        m = model.clone()
        current_options = m.model_options.get("transformer_options", {}).copy()

        if slice_size == 0:
            # Nettoyage si Auto
            if "attention_slice_size" in current_options:
                del current_options["attention_slice_size"]
        else:
            current_options["memory_efficient_attention"] = True
            current_options["attention_slice_size"] = slice_size
            
        m.model_options["transformer_options"] = current_options
        return (m,)

NODE_CLASS_MAPPINGS = {
    "Wan_Hardware_Accelerator": Wan_Hardware_Accelerator,
    "Wan_Attention_Slicer": Wan_Attention_Slicer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Hardware_Accelerator": "Wan Hardware Accelerator (Local Cast)",
    "Wan_Attention_Slicer": "Wan Attention Strategy"
}
