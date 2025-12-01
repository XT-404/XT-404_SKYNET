import torch
import comfy.model_management as mm
from comfy.sd import VAE
import torch.nn.functional as F

# ==============================================================================
# WAN ARCHITECT: RADICAL CLEANUP (V21)
# ==============================================================================

class WanSubState:
    def __init__(self):
        self.prev_latent = None
        self.prev_output = None
        self.skipped_steps = 0
        self.step_counter = 0

class WanState:
    def __init__(self):
        self.flows = {} 
        self.autocast_failed_once = False 

class Wan_TeaCache_Patch:
    """
    Wan Turbo (TeaCache).
    Module conservé intact comme demandé.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_tea_cache": ("BOOLEAN", {"default": True}),
                "rel_l1_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.6, "step": 0.001}),
                "start_step_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "force_autocast": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("turbo_model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "ComfyWan_Architect/Performance"

    def apply_teacache(self, model, enable_tea_cache, rel_l1_threshold, start_step_percent, force_autocast):
        if not enable_tea_cache: return (model,)
        
        m = model.clone()
        m.wan_teacache_state = WanState()
        device = mm.get_torch_device()
        
        # Détection auto du support BF16
        dtype = torch.float16
        if force_autocast and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        def teacache_wrapper(model_function, params):
            input_x = params.get("input")
            timestep = params.get("timestep")
            c = params.get("c", {})
            
            # Gestion Flow ID simple
            try: flow_id = c.get("context", c.get("y", torch.tensor(0))).data_ptr()
            except: flow_id = "global"

            if flow_id not in m.wan_teacache_state.flows: 
                m.wan_teacache_state.flows[flow_id] = WanSubState()
            state = m.wan_teacache_state.flows[flow_id]

            # Helper execution
            def run(x, t, **k):
                if not m.wan_teacache_state.autocast_failed_once:
                    try: 
                        with torch.autocast(device.type, dtype=dtype): 
                            return model_function(x, t, **k)
                    except: 
                        m.wan_teacache_state.autocast_failed_once = True
                        return model_function(x, t, **k)
                return model_function(x, t, **k)

            # Logique de saut
            if state.prev_latent is None or state.step_counter < 2 or input_x.shape != state.prev_latent.shape:
                state.prev_latent = input_x.detach()
                state.prev_output = run(input_x, timestep, **c)
                state.step_counter += 1
                return state.prev_output

            # Calcul différence
            dims = input_x.shape
            stride = 4 if (dims[-1]*dims[-2]) > 262144 else 2
            try:
                diff = (input_x[...,::stride,::stride] - state.prev_latent[...,::stride,::stride]).abs().mean()
                if diff < rel_l1_threshold:
                    state.skipped_steps += 1
                    state.step_counter += 1
                    return state.prev_output
            except: pass

            state.prev_output = run(input_x, timestep, **c)
            state.prev_latent = input_x.detach()
            state.step_counter += 1
            return state.prev_output

        m.set_model_unet_function_wrapper(teacache_wrapper)
        return (m,)

class Wan_Hybrid_VRAM_Guard:
    """
    OMEGA V21: PASS-THROUGH (GHOST MODE)
    Ce nœud ne fait plus RIEN. Il appelle simplement le décodage standard de ComfyUI.
    Il garde les noms de paramètres pour ne pas casser ton workflow existant.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
                # Paramètres conservés pour compatibilité UI (Ignorés par le code)
                "tile_size_spatial": ("INT", {"default": 512, "min": 256, "max": 4096}),
                "temporal_chunk_size": ("INT", {"default": 8, "min": 1, "max": 64}), 
                "enable_cpu_offload": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_standard"
    CATEGORY = "ComfyWan_Architect/Performance"
    
    def decode_standard(self, vae, samples, tile_size_spatial, temporal_chunk_size, enable_cpu_offload):
        # On ignore totalement tile_size, chunk_size, etc.
        # On utilise la méthode standard de ComfyUI qui gère tout toute seule.
        return (vae.decode(samples["samples"]),)

# MAPPINGS
NODE_CLASS_MAPPINGS = {
    "Wan_TeaCache_Patch": Wan_TeaCache_Patch,
    "Wan_Hybrid_VRAM_Guard": Wan_Hybrid_VRAM_Guard
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_TeaCache_Patch": "Wan Turbo (TeaCache Omega)",
    "Wan_Hybrid_VRAM_Guard": "Wan Decode (Standard Native)"
}
