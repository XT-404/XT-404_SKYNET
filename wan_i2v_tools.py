import torch
import torch.nn.functional as F
import comfy.model_management as mm
import gc
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
    """V8.1: Cache Vision Secured (Robust Hash)."""
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
        # CORRECTIF HASH V8.1 : Réduction du stride (32->16) et utilisation de bytes bruts
        # pour éviter les collisions sur les images similaires (ex: variations subtiles).
        # On inclut aussi la shape dans le hash pour distinguer les résolutions.
        
        # Stride 16 est un bon compromis pour 1080p/4K
        stride = 16
        if image.device.type == 'cuda':
            # Extraction d'un sous-ensemble significatif
            subset = image[:, ::stride, ::stride, :]
            sig_cpu = subset.flatten().cpu().numpy().tobytes()
        else:
            sig_cpu = image[:, ::stride, ::stride, :].flatten().numpy().tobytes()

        # Construction clé unique : Hash du contenu + Shape + ID modèle
        base_hash = hashlib.md5(sig_cpu).hexdigest()
        img_hash = f"{base_hash}_{image.shape}_{id(clip_vision)}"
        
        if img_hash in self._cache:
            print(f">> [Wan I2V] Cache Hit (Vision Encodings).")
            self._cache.move_to_end(img_hash)
            data = self._cache[img_hash]
            return (safe_to(data, mm.get_torch_device()), image)
        
        print(f">> [Wan I2V] Encoding Vision...")
        output = clip_vision.encode_image(image)
        cpu_output = safe_to(output, "cpu")
        
        self._cache[img_hash] = cpu_output
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        
        if aggressive_offload:
            try:
                if hasattr(clip_vision, "patcher"): clip_vision.patcher.model.to("cpu")
                elif hasattr(clip_vision, "model"): clip_vision.model.to("cpu")
            except: pass
            mm.soft_empty_cache()
            
        return (output, image)

class Wan_Resolution_Savant:
    """
    V10 PLATINUM: Gestion intelligente de la résolution.
    - Supporte Lanczos (CPU/PIL) pour la qualité maximale.
    - Supporte Area (GPU) pour le downscaling rapide et propre.
    - Force l'alignement divisible par 16 (requis par Wan).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_dimension": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "divisible_by": ("INT", {"default": 16}),
                "resampling_mode": (["lanczos", "bicubic", "area", "bilinear", "nearest"], {"default": "lanczos"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "optimize_resolution"
    CATEGORY = "ComfyWan_Architect/I2V"

    def optimize_resolution(self, image, max_dimension, divisible_by, resampling_mode):
        B, H, W, C = image.shape
        
        # Calcul de l'échelle pour ne pas dépasser max_dimension
        scale = min(max_dimension / W, max_dimension / H, 1.0)
        new_w = int(W * scale)
        new_h = int(H * scale)
        
        # Ajustement strict pour la divisibilité (ex: divisible par 16)
        target_w = max(round(new_w / divisible_by) * divisible_by, divisible_by)
        target_h = max(round(new_h / divisible_by) * divisible_by, divisible_by)
        
        # Si dimensions identiques, on renvoie l'original
        if target_w == W and target_h == H: return (image,)
        
        # --- LOGIQUE DE DÉCISION CPU vs GPU ---
        
        # 1. Image énorme sur CPU (risque OOM GPU lors du resize)
        is_huge_cpu = (target_w * target_h > 2048*2048) and (image.device.type == "cpu")
        
        # 2. Mode Lanczos demandé (PyTorch ne l'a pas, PIL est requis)
        force_pil_quality = (resampling_mode == "lanczos")

        if is_huge_cpu or force_pil_quality:
            # --- CHEMIN PIL (CPU) ---
            results = []
            
            # Mapping des modes pour PIL
            pil_mode = Image.BICUBIC # Valeur par défaut
            if resampling_mode == "lanczos": 
                pil_mode = Image.LANCZOS
            elif resampling_mode == "nearest": 
                pil_mode = Image.NEAREST
            elif resampling_mode == "bilinear": 
                pil_mode = Image.BILINEAR
            elif resampling_mode == "area":
                # 'BOX' est l'équivalent PIL de 'Area' pour le downscaling
                pil_mode = Image.BOX 

            for b in range(B):
                tensor_img = image[b]
                # Si l'image est sur GPU, on la ramène sur CPU
                if tensor_img.device.type != 'cpu':
                    tensor_img = tensor_img.cpu()
                
                # Conversion Tensor -> Numpy -> PIL
                np_img = (tensor_img.numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img)
                
                # Resize
                resized_pil = pil_img.resize((target_w, target_h), resample=pil_mode)
                
                # Retour vers Tensor
                out_tensor = torch.from_numpy(np.array(resized_pil)).float() / 255.0
                results.append(out_tensor)
            
            final_tensor = torch.stack(results)
            
            # Si l'entrée était sur GPU, on renvoie le résultat sur GPU
            if image.device.type == 'cuda':
                final_tensor = final_tensor.to(image.device)
            return (final_tensor,)

        # --- CHEMIN GPU (Accéléré) ---
        # Utilisé pour bicubic, bilinear, area, nearest
        
        device = mm.get_torch_device()
        # PyTorch attend (Batch, Channel, Height, Width) -> Permutation
        img_permuted = image.to(device).permute(0, 3, 1, 2)
        
        # Optimisation FP16 pour le resize (plus rapide sur RTX)
        if device.type == 'cuda':
             img_permuted = img_permuted.to(memory_format=torch.channels_last).half()

        # Configuration des flags PyTorch
        align_corners = False
        antialias = True # Important pour éviter le scintillement
        
        if resampling_mode == "nearest":
            align_corners = None
            antialias = False
        elif resampling_mode == "area":
            # 'area' est adaptatif, pas d'align_corners
            align_corners = None
            antialias = False 
        
        img_resized = F.interpolate(
            img_permuted, 
            size=(target_h, target_w), 
            mode=resampling_mode, 
            align_corners=align_corners,
            antialias=antialias 
        )
        
        # Retour au format standard Comfy (Batch, Height, Width, Channel) en FP32
        result = img_resized.float().permute(0, 2, 3, 1).contiguous()
        
        return (result.to(image.device),)
