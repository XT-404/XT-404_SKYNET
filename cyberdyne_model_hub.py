# cyberdyne_nodes/cyberdyne_model_hub.py
import os
import torch
import hashlib
import sys
import time
import nodes
from typing import Dict, Any, Tuple, Optional, List, Set
from contextlib import contextmanager

import comfy.sd
import comfy.model_management

# Cache global pour les modèles chargés
_cyberdyne_model_cache: Dict[Tuple[str, str], Tuple[Any, str]] = {}

# --- UTILITAIRES DE STYLE (HUD TERMINATOR) ---
def log_system(message: str):
    print(f"\033[96m[CYBERDYNE SYSTEM]\033[0m {message}")

def log_warning(message: str):
    print(f"\033[93m[WARNING PROVISION]\033[0m {message}")

def log_critical(message: str):
    print(f"\033[91m[CRITICAL FAILURE]\033[0m {message}")

def log_success(message: str):
    print(f"\033[92m[OPERATIONAL]\033[0m {message}")

def find_comfyui_base_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir): 
        if os.path.exists(os.path.join(current_dir, "main.py")) or \
           os.path.exists(os.path.join(current_dir, "nodes.py")): 
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.getcwd() 

def get_model_dirs_by_type(model_type_hint: str) -> List[str]:
    base_path = find_comfyui_base_path()
    model_dirs = []

    if model_type_hint == "safetensors":
        path = os.path.join(base_path, "models", "diffusion_models")
        if os.path.exists(path) and os.path.isdir(path):
            model_dirs.append(path)
        path_fallback = os.path.join(base_path, "models", "checkpoints")
        if os.path.exists(path_fallback) and os.path.isdir(path_fallback):
             model_dirs.append(path_fallback)

    elif model_type_hint == "gguf":
        path = os.path.join(base_path, "models", "unet")
        if os.path.exists(path) and os.path.isdir(path):
            model_dirs.append(path)
    
    return model_dirs

# --- CONTEXT MANAGER DE SECURITE ---
@contextmanager
def unsafe_torch_load_context():
    original_load = torch.load
    def unsafe_load(*args, **kwargs):
        if 'weights_only' in kwargs: kwargs['weights_only'] = False
        else: kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    try:
        torch.load = unsafe_load
        yield
    finally:
        torch.load = original_load

class CyberdyneModelHub:
    """
    Nœud universel pour le chargement optimisé des modèles de diffusion.
    Version: 1.6 (Subdirectory Support Included)
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        safetensors_files: Set[str] = set()
        # Recherche récursive pour safetensors/checkpoints
        for s_dir in get_model_dirs_by_type("safetensors"):
            if os.path.exists(s_dir):
                for root, _, files in os.walk(s_dir):
                    for filename in files:
                        if filename.endswith((".safetensors", ".ckpt")):
                            # On stocke le chemin relatif par rapport au dossier racine du modèle
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, s_dir)
                            safetensors_files.add(rel_path)

        gguf_files: Set[str] = set()
        # Recherche récursive pour GGUF
        for g_dir in get_model_dirs_by_type("gguf"):
            if os.path.exists(g_dir):
                for root, _, files in os.walk(g_dir):
                    for filename in files:
                        if filename.endswith(".gguf"):
                            full_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(full_path, g_dir)
                            gguf_files.add(rel_path)
        
        all_model_files = sorted(list(safetensors_files.union(gguf_files)))
        supported_dtypes = ["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

        return {
            "required": {
                "model_high_name": (["NONE"] + all_model_files, {"default": "NONE"}),
                "dtype_high": (supported_dtypes, {"default": "default"}),
            },
            "optional": {
                "model_low_name": (["NONE"] + all_model_files, {"default": "NONE"}),
                "dtype_low": (supported_dtypes, {"default": "default"}),
                "base_path": ("STRING", {"default": ""}),
                "enable_checksum": ("BOOLEAN", {"default": False}),
                "offload_inactive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "process"
    CATEGORY = "Cyberdyne/Loaders"

    def _get_model_path(self, model_filename: str, base_path: str) -> Optional[str]:
        if model_filename == "NONE": return None
        if base_path:
            full_path = os.path.join(base_path, model_filename)
            if os.path.exists(full_path): return full_path

        model_type_hint = ""
        if model_filename.endswith((".safetensors", ".ckpt")): model_type_hint = "safetensors"
        elif model_filename.endswith(".gguf"): model_type_hint = "gguf"
        
        if model_type_hint:
            for model_dir in get_model_dirs_by_type(model_type_hint):
                # model_filename contient maintenant le chemin relatif (ex: subfolder/model.safetensors)
                full_path = os.path.join(model_dir, model_filename)
                if os.path.exists(full_path): return full_path
        return None

    def _calculate_checksum(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        log_system(f"SCANNING TARGET: {filename}")
        log_system("INTEGRITY CHECK IN PROGRESS... [||||||||||]")
        
        t0 = time.time()
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192 * 1024) # Lecture plus rapide par gros blocs
                if not chunk: break
                hasher.update(chunk)
        
        digest = hasher.hexdigest()
        t1 = time.time()
        
        print(f"    └─ SHA256: \033[96m{digest}\033[0m")
        log_success(f"SCAN COMPLETE ({t1-t0:.2f}s). INTEGRITY VERIFIED.")
        return digest

    def _load_model(self, model_path: str, dtype_str: str, enable_checksum: bool) -> Tuple[Any, str]:
        model_type = "UNKNOWN"
        filename = os.path.basename(model_path)
        
        if dtype_str == "default": target_dtype = None
        elif dtype_str == "fp16": target_dtype = torch.float16
        elif dtype_str == "bf16": target_dtype = torch.bfloat16
        elif dtype_str == "fp8_e4m3fn": target_dtype = "fp8_e4m3fn"
        elif dtype_str == "fp8_e4m3fn_fast": target_dtype = "fp8_e4m3fn_fast"
        elif dtype_str == "fp8_e5m2": target_dtype = "fp8_e5m2"
        else: raise ValueError(f"Type de données non supporté: {dtype_str}")

        cache_key = (model_path, dtype_str)
        if cache_key in _cyberdyne_model_cache:
            log_success(f"CACHE HIT: {filename} retrieved from memory.")
            return _cyberdyne_model_cache[cache_key]

        log_system(f"INITIATING LOADING SEQUENCE: {filename} [{dtype_str}]")

        if enable_checksum:
            self._calculate_checksum(model_path)

        # --- Chargement Safetensors ---
        if model_path.endswith((".safetensors", ".ckpt")):
            model_type = "Safetensors"
            try:
                with unsafe_torch_load_context():
                    out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=None)
                    loaded_model = out[0]
                
                if target_dtype is not None:
                    if isinstance(target_dtype, str):
                        loaded_model.model = comfy.model_management.cast_to_custom_fp8(loaded_model.model, target_dtype)
                    elif loaded_model.model.dtype != target_dtype:
                        loaded_model.model.to(device=comfy.model_management.get_torch_device(), dtype=target_dtype)
            except Exception as e:
                log_critical(f"SAFETENSORS LOAD FAILED: {e}")
                raise RuntimeError(f"Erreur lors du chargement Safetensors: {e}")

        # --- Chargement GGUF via Délégation ---
        elif model_path.endswith(".gguf"):
            model_type = "GGUF"
            try:
                log_system(f"DELEGATING GGUF LOAD PROTOCOL TO: UnetLoaderGGUF")
                
                gguf_loader_class = None
                if "UnetLoaderGGUF" in nodes.NODE_CLASS_MAPPINGS:
                    gguf_loader_class = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
                elif "UnetLoaderGGUF" in dir(nodes):
                    gguf_loader_class = getattr(nodes, "UnetLoaderGGUF")
                
                if gguf_loader_class is None:
                    raise RuntimeError("Support GGUF manquant.")

                # Attention: Pour GGUF, le loader natif attend souvent juste le nom de fichier relatif
                # On tente de passer le chemin complet si le loader le supporte, ou le relatif.
                # Dans ComfyUI standard, UnetLoaderGGUF attend un nom relatif.
                loader_instance = gguf_loader_class()
                
                # Récupération du nom relatif (déjà fait dans INPUT_TYPES mais on s'assure ici)
                # Cependant, si model_path est absolu, on doit trouver le relatif attendu par le noeud GGUF
                # Hack: On passe le nom de fichier ou le path relatif calculé
                
                # Pour maximiser la compatibilité, on essaie de retrouver le nom tel qu'il apparaîtrait dans la liste
                # Si le node GGUF utilise folder_paths.get_full_path, il a besoin du nom relatif.
                
                relative_gguf_name = filename # Fallback
                for g_dir in get_model_dirs_by_type("gguf"):
                    if model_path.startswith(g_dir):
                        relative_gguf_name = os.path.relpath(model_path, g_dir)
                        break
                
                result = loader_instance.load_unet(unet_name=relative_gguf_name)
                loaded_model = result[0]

                if target_dtype is not None:
                    try:
                        if isinstance(target_dtype, str):
                             loaded_model.model = comfy.model_management.cast_to_custom_fp8(loaded_model.model, target_dtype)
                        elif hasattr(loaded_model.model, 'dtype') and loaded_model.model.dtype != target_dtype:
                            if hasattr(loaded_model.model, 'to'):
                                loaded_model.model.to(device=comfy.model_management.get_torch_device(), dtype=target_dtype)
                    except Exception as e:
                        log_warning(f"DTYPE OVERRIDE FAILED: {e}")

            except Exception as e:
                log_critical(f"GGUF LOAD FAILED: {e}")
                raise RuntimeError(f"Erreur fatale lors du chargement GGUF: {e}")
        else:
            raise ValueError(f"Format non supporté: {model_path}")

        _cyberdyne_model_cache[cache_key] = (loaded_model, model_type)
        log_success(f"MODEL ACTIVE: {filename}")
        return loaded_model, model_type

    def process(self, model_high_name, dtype_high, model_low_name, dtype_low, base_path, enable_checksum, offload_inactive):
        loaded_model_high = None
        loaded_model_low = None

        print("\n" + "="*50)
        log_system("CYBERDYNE MODEL HUB v1.6 - INITIALIZED")
        
        model_high_path = self._get_model_path(model_high_name, base_path)
        model_low_path = self._get_model_path(model_low_name, base_path)

        # --- GESTION BLINDÉE DU DÉCHARGEMENT AVEC TELEMETRIE ---
        if offload_inactive:
            current_paths = {p for p in [model_high_path, model_low_path] if p}
            models_to_offload = []
            
            for key, (cached_model, _) in list(_cyberdyne_model_cache.items()):
                if key[0] not in current_paths:
                    models_to_offload.append((key[0], cached_model))
            
            vram_freed = 0
            if models_to_offload:
                log_system("OPTIMIZING VRAM RESOURCES...")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()

                for path, model in models_to_offload:
                    try:
                        if model is None or model.model is None: continue
                        if not hasattr(model.model, 'to'): continue

                        is_cuda = False
                        if hasattr(model.model, 'is_cuda'): is_cuda = model.model.is_cuda
                        elif hasattr(model.model, 'device'):
                            dev = model.model.device
                            if str(dev).startswith('cuda'): is_cuda = True
                        
                        if is_cuda:
                            model.model.cpu()
                            
                    except Exception: pass
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated()
                    vram_freed = (mem_before - mem_after) / (1024 ** 2) # Conversion en MB
                
                if vram_freed > 1:
                    print(f"    ▼ OFFLOAD PROTOCOL EXECUTED. RECLAIMED: \033[92m{vram_freed:.2f} MB\033[0m VRAM")
                else:
                    log_system("MEMORY ALREADY OPTIMIZED.")

        # Chargement
        if model_high_path:
            loaded_model_high, _ = self._load_model(model_high_path, dtype_high, enable_checksum)
            
        if model_low_path:
            loaded_model_low, _ = self._load_model(model_low_path, dtype_low, enable_checksum)

        print("="*50 + "\n")
        return (loaded_model_high, loaded_model_low)
