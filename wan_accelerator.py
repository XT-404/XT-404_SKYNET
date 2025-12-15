import torch
import torch.nn as nn
import comfy.model_management as mm
import torch.nn.functional as F

class Wan_Hardware_Accelerator:
    """
    OMEGA EDITION V4: Contextual Precision Switching.
    SOLUTION DÉFINITIVE "ANTI-BURN".
    
    Le problème : TF32 "brûle" les images en ratant les calculs de Normalisation.
    La solution : Ce code active TF32 pour tout le modèle (Vitesse), 
    MAIS le désactive chirurgicalement juste pour les couches Norm (Qualité).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                # On laisse True par défaut car le patch corrige le défaut visuel
                "enable_tf32": ("BOOLEAN", {"default": True, "tooltip": "Garde la vitesse du TF32 sans l'effet brûlé grâce au correctif hybride."}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True, "tooltip": "Optimise les convolutions."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("accelerated_model",)
    FUNCTION = "apply_acceleration"
    CATEGORY = "ComfyWan_Architect/Performance"

    def _stabilize_precision(self, module):
        """
        Installe un interrupteur intelligent sur chaque couche sensible.
        """
        for name, child in module.named_children():
            # Cible : GroupNorm et LayerNorm (responsables du contraste/luminosité)
            if isinstance(child, (nn.GroupNorm, nn.LayerNorm)):
                if not hasattr(child, "original_forward"):
                    child.original_forward = child.forward

                # Voici le secret : Le "Context Switch"
                def secure_forward(x, *args, _layer=child, **kwargs):
                    # 1. On sauvegarde l'état actuel de TF32
                    prev_matmul = torch.backends.cuda.matmul.allow_tf32
                    prev_cudnn = torch.backends.cudnn.allow_tf32
                    
                    # 2. ON COUPE TF32 (Pour avoir la qualité parfaite)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    
                    try:
                        # 3. On force les données en Haute Précision (FP32)
                        # Le calcul se fait ici en pure précision mathématique
                        x_32 = x.float()
                        res = _layer.original_forward(x_32, *args, **kwargs)
                        
                        # 4. On revient au format d'origine (FP16/BF16) pour la suite
                        return res.type(x.dtype)
                    finally:
                        # 5. ON RALLUME TF32 (Pour la vitesse du reste du modèle)
                        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
                        torch.backends.cudnn.allow_tf32 = prev_cudnn
                
                child.forward = secure_forward
                
            else:
                self._stabilize_precision(child)

    def apply_acceleration(self, model, enable_tf32, cudnn_benchmark):
        m = model.clone()

        # 1. Activation Global du Moteur TF32
        if torch.cuda.is_available():
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"⚡ [Wan Accel] TF32 ENGINE STARTED (High Speed).")
                
                # 2. Installation du correctif Anti-Burn
                if hasattr(m.model, "diffusion_model"):
                    print(f"🛡️ [Wan Accel] Installing Contextual Switches on Norm Layers...")
                    self._stabilize_precision(m.model.diffusion_model)
                    print(f"✅ [Wan Accel] Anti-Burn Protection Active: Sensitive layers will run in Native FP32.")
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                print(f"🐢 [Wan Accel] TF32 DISABLED globally.")
        
        if cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        return (m,)

class Wan_Attention_Slicer:
    """
    OMEGA EDITION: Smart Attention Management.
    """
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
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        current_options = m.model_options.get("transformer_options", {}).copy()

        if slice_size == 0:
            if has_sdpa:
                if "attention_slice_size" in current_options:
                    del current_options["attention_slice_size"]
            else:
                current_options["attention_slice_size"] = 8
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
    "Wan_Hardware_Accelerator": "Wan Hardware Accelerator (Anti-Burn V4)",
    "Wan_Attention_Slicer": "Wan Attention Strategy (Omega)"
}
