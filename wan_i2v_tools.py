import torch
import torch.nn.functional as F
import comfy.model_management as mm
import hashlib
from collections import OrderedDict
import numpy as np
from PIL import Image

def safe_to(data, device):
    if hasattr(data, "to"): return data.to(device)
    elif isinstance(data, (list, tuple)): return type(data)([safe_to(x, device) for x in data])
    elif isinstance(data, dict): return {k: safe_to(v, device) for k, v in data.items()}
    return data

class Wan_Vision_OneShot_Cache:
    """V8.2 OMEGA: Hash robuste incluant la shape."""
    _cache = OrderedDict()
    CACHE_LIMIT = 5 

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "aggressive_offload": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("CLIP_VISION_OUTPUT", "IMAGE")
    RETURN_NAMES = ("vision_output", "passthrough_image")
    FUNCTION = "encode_vision_oneshot"
    CATEGORY = "ComfyWan_Architect/I2V"

    def encode_vision_oneshot(self, clip_vision, image, aggressive_offload):
        # Hashage rapide sur sous-échantillon (Stride 16)
        stride = 16
        if image.device.type == 'cuda':
            sig_cpu = image[:, ::stride, ::stride, :].flatten().cpu().numpy().tobytes()
        else:
            sig_cpu = image[:, ::stride, ::stride, :].flatten().numpy().tobytes()

        base_hash = hashlib.md5(sig_cpu).hexdigest()
        img_hash = f"{base_hash}_{image.shape}_{id(clip_vision)}"
        
        if img_hash in self._cache:
            print(f"👁️ [Wan I2V] Vision Cache Hit.")
            self._cache.move_to_end(img_hash)
            data = self._cache[img_hash]
            return (safe_to(data, mm.get_torch_device()), image)
        
        print(f"👁️ [Wan I2V] Encoding Vision...")
        output = clip_vision.encode_image(image)
        cpu_output = safe_to(output, "cpu")
        
        self._cache[img_hash] = cpu_output
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        
        if aggressive_offload:
            mm.soft_empty_cache()
            
        return (output, image)

class Wan_Resolution_Savant:
    """
    OMEGA EDITION: True FP32 Resampling.
    GARANTIE : Le redimensionnement s'effectue en FP32 pour éviter le banding et l'aliasing.
    NOTE: 'lanczos' utilise le CPU (PIL) pour une qualité maximale. Les autres utilisent le GPU.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_dimension": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "divisible_by": ("INT", {"default": 16, "tooltip": "Wan requiert souvent 16 ou 32."}),
                # RESTAURATION COMPATIBILITÉ : Noms simples pour ne pas casser le workflow
                "resampling_mode": (["lanczos", "bicubic", "area", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "optimize_resolution"
    CATEGORY = "ComfyWan_Architect/I2V"

    def optimize_resolution(self, image, max_dimension, divisible_by, resampling_mode):
        B, H, W, C = image.shape
        
        scale = min(max_dimension / W, max_dimension / H, 1.0)
        new_w = int(W * scale)
        new_h = int(H * scale)
        
        target_w = max(round(new_w / divisible_by) * divisible_by, divisible_by)
        target_h = max(round(new_h / divisible_by) * divisible_by, divisible_by)
        
        if target_w == W and target_h == H: return (image,)
        
        # Mode CPU/PIL (Lanczos - Qualité Ultime)
        # La logique détecte "lanczos" tout court maintenant
        if resampling_mode == "lanczos":
            results = []
            for b in range(B):
                tensor_img = image[b]
                if tensor_img.device.type != 'cpu': tensor_img = tensor_img.cpu()
                
                np_img = (tensor_img.numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img)
                resized_pil = pil_img.resize((target_w, target_h), resample=Image.LANCZOS)
                
                out_tensor = torch.from_numpy(np.array(resized_pil)).float() / 255.0
                results.append(out_tensor)
            
            final_tensor = torch.stack(results)
            if image.device.type == 'cuda': final_tensor = final_tensor.to(image.device)
            return (final_tensor,)

        # Mode GPU (Optimisé FP32 OMEGA)
        device = mm.get_torch_device()
        
        # LOI 2 OMEGA: Interpolation en FP32 OBLIGATOIRE
        img_permuted = image.to(device, dtype=torch.float32).permute(0, 3, 1, 2)
        
        align = False if resampling_mode in ["bicubic", "bilinear"] else None
        antialias = True if resampling_mode in ["bicubic", "bilinear"] else False
        
        img_resized = F.interpolate(
            img_permuted, 
            size=(target_h, target_w), 
            mode=resampling_mode, 
            align_corners=align,
            antialias=antialias 
        )
        
        result = img_resized.permute(0, 2, 3, 1).contiguous()
        return (result,)

NODE_CLASS_MAPPINGS = {
    "Wan_Vision_OneShot_Cache": Wan_Vision_OneShot_Cache,
    "Wan_Resolution_Savant": Wan_Resolution_Savant
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Vision_OneShot_Cache": "Wan Vision Cache (Omega)",
    "Wan_Resolution_Savant": "Wan Resolution (Omega FP32)"
}
