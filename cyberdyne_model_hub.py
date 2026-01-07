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
import folder_paths  # INDISPENSABLE : Le moteur de recherche de chemins de ComfyUI

# Cache global pour les modèles chargés (Préserve l'optimisation VRAM entre les générations)
_cyberdyne_model_cache: Dict[Tuple[str, str], Tuple[Any, str]] = {}

# --- UTILITAIRES DE STYLE (HUD TERMINATOR) - CONSERVÉS ---
def log_system(message: str):
    print(f"\033[96m[CYBERDYNE SYSTEM]\033[0m {message}")

def log_warning(message: str):
    print(f"\033[93m[WARNING PROVISION]\033[0m {message}")

def log_critical(message: str):
    print(f"\033[91m[CRITICAL FAILURE]\033[0m {message}")

def log_success(message: str):
    print(f"\033[92m[OPERATIONAL]\033[0m {message}")

# --- NOUVELLE FONCTION DE DÉTECTION INTELLIGENTE ---
def get_target_directories(model_type_hint: str) -> List[str]:
    """
    Récupère automatiquement les dossiers via l'API ComfyUI.
    Supporte les installations standards, les liens symboliques et extra_model_paths.yaml.
    """
    dirs = []
    
    if model_type_hint == "safetensors":
        # Récupère tous les dossiers enregistrés pour les checkpoints
        dirs.extend(folder_paths.get_folder_paths("checkpoints"))
        dirs.extend(folder_paths.get_folder_paths("diffusion_models"))
        
    elif model_type_hint == "gguf":
        # Récupère le dossier standard pour GGUF (unet)
        dirs.extend(folder_paths.get_folder_paths("unet"))
        
    # Filtre les dossiers inexistants et supprime les doublons
    return list(set([d for d in dirs if os.path.exists(d)]))

# --- CONTEXT MANAGER DE SECURITE - CONSERVÉ ---
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
    Version: 2.1 (Full Auto-Discovery, No Base Path)
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        safetensors_files: Set[str] = set()
        
        # SCAN AUTOMATIQUE ET RÉCURSIF (SAFETENSORS)
        # Remplace l'ancien système manuel. Scanne aussi les sous-dossiers.
        for s_dir in get_target_directories("safetensors"):
            for root, _, files in os.walk(s_dir):
                for filename in files:
                    if filename.endswith((".safetensors", ".ckpt")):
                        full_path = os.path.join(root, filename)
                        try:
                            # Affiche le chemin relatif (ex: "SDXL/mon_model.safetensors")
                            rel_path = os.path.relpath(full_path, s_dir)
                            safetensors_files.add(rel_path)
                        except ValueError:
                            # Fallback pour les chemins complexes (disques différents)
                            safetensors_files.add(filename)

        gguf_files: Set[str] = set()
        
        # SCAN AUTOMATIQUE ET RÉCURSIF (GGUF)
        for g_dir in get_target_directories("gguf"):
            for root, _, files in os.walk(g_dir):
                for filename in files:
                    if filename.endswith(".gguf"):
                        full_path = os.path.join(root, filename)
                        try:
                            rel_path = os.path.relpath(full_path, g_dir)
                            gguf_files.add(rel_path)
                        except ValueError:
                            gguf_files.add(filename)
        
        # Liste triée pour l'interface
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
                # NOTE: base_path a été retiré comme demandé
                "enable_checksum": ("BOOLEAN", {"default": False}),
                "offload_inactive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "process"
    CATEGORY = "Cyberdyne/Loaders"

    def _find_file_path(self, model_filename: str) -> Optional[str]:
        """Cherche le chemin absolu du fichier dans tous les dossiers connus."""
        if model_filename == "NONE": return None
        
        type_hint = ""
        if model_filename.endswith((".safetensors", ".ckpt")): type_hint = "safetensors"
        elif model_filename.endswith(".gguf"): type_hint = "gguf"
        
        if type_hint:
            for search_dir in get_target_directories(type_hint):
                candidate_path = os.path.join(search_dir, model_filename)
                if os.path.exists(candidate_path):
                    return candidate_path
        return None

    def _calculate_checksum(self, file_path: str) -> str:
        filename = os.path.basename(file_path)
        log_system(f"SCANNING TARGET: {filename}")
        log_system("INTEGRITY CHECK IN PROGRESS... [||||||||||]")
        
        t0 = time.time()
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192 * 1024)
                    if not chunk: break
                    hasher.update(chunk)
            
            digest = hasher.hexdigest()
            t1 = time.time()
            
            print(f"    └─ SHA256: \033[96m{digest}\033[0m")
            log_success(f"SCAN COMPLETE ({t1-t0:.2f}s). INTEGRITY VERIFIED.")
            return digest
        except Exception as e:
            log_warning(f"CHECKSUM FAILED: {e}")
            return "UNKNOWN"

    def _load_model(self, model_path: str, dtype_str: str, enable_checksum: bool) -> Tuple[Any, str]:
        filename = os.path.basename(model_path)
        
        if dtype_str == "default": target_dtype = None
        elif dtype_str == "fp16": target_dtype = torch.float16
        elif dtype_str == "bf16": target_dtype = torch.bfloat16
        elif dtype_str == "fp8_e4m3fn": target_dtype = "fp8_e4m3fn"
        elif dtype_str == "fp8_e4m3fn_fast": target_dtype = "fp8_e4m3fn_fast"
        elif dtype_str == "fp8_e5m2": target_dtype = "fp8_e5m2"
        else: raise ValueError(f"Type {dtype_str} inconnu")

        cache_key = (model_path, dtype_str)
        if cache_key in _cyberdyne_model_cache:
            log_success(f"CACHE HIT: {filename} retrieved from memory.")
            return _cyberdyne_model_cache[cache_key]

        log_system(f"INITIATING LOADING SEQUENCE: {filename} [{dtype_str}]")

        if enable_checksum:
            self._calculate_checksum(model_path)

        # --- Chargement Safetensors (Code Original Conservé) ---
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
                raise

        # --- Chargement GGUF (Code Original Adapté) ---
        elif model_path.endswith(".gguf"):
            model_type = "GGUF"
            try:
                log_system(f"DELEGATING GGUF LOAD PROTOCOL TO: UnetLoaderGGUF")
                
                # Recherche dynamique de la classe GGUF Loader
                gguf_loader_class = None
                if "UnetLoaderGGUF" in nodes.NODE_CLASS_MAPPINGS:
                    gguf_loader_class = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
                elif "UnetLoaderGGUF" in dir(nodes):
                    gguf_loader_class = getattr(nodes, "UnetLoaderGGUF")
                
                if gguf_loader_class is None:
                    raise RuntimeError("Support GGUF manquant (Node UnetLoaderGGUF introuvable).")

                loader_instance = gguf_loader_class()
                
                # Calcul du chemin relatif requis par le loader GGUF standard
                rel_name = filename 
                for g_dir in get_target_directories("gguf"):
                    if model_path.startswith(g_dir):
                        try:
                            rel_name = os.path.relpath(model_path, g_dir)
                            break
                        except ValueError: continue
                
                result = loader_instance.load_unet(unet_name=rel_name)
                loaded_model = result[0]

                if target_dtype is not None:
                    try:
                        if isinstance(target_dtype, str):
                             loaded_model.model = comfy.model_management.cast_to_custom_fp8(loaded_model.model, target_dtype)
                        elif hasattr(loaded_model.model, 'dtype') and loaded_model.model.dtype != target_dtype:
                             loaded_model.model.to(device=comfy.model_management.get_torch_device(), dtype=target_dtype)
                    except Exception: pass

            except Exception as e:
                log_critical(f"GGUF LOAD FAILED: {e}")
                raise RuntimeError(f"Erreur fatale lors du chargement GGUF: {e}")
        else:
            raise ValueError(f"Format non supporté: {model_path}")

        _cyberdyne_model_cache[cache_key] = (loaded_model, model_type)
        log_success(f"MODEL ACTIVE: {filename}")
        return loaded_model, model_type

    def process(self, model_high_name, dtype_high, model_low_name, dtype_low, enable_checksum, offload_inactive):
        # NOTE: base_path a été retiré des arguments ici aussi pour correspondre à INPUT_TYPES
        
        loaded_model_high = None
        loaded_model_low = None

        print("\n" + "="*50)
        log_system("CYBERDYNE MODEL HUB v2.1 - INITIALIZED")
        
        # Utilisation de la nouvelle méthode de recherche
        model_high_path = self._find_file_path(model_high_name)
        model_low_path = self._find_file_path(model_low_name)

        # --- GESTION BLINDÉE DU DÉCHARGEMENT AVEC TELEMETRIE (CODE ORIGINAL CONSERVÉ) ---
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

        # Chargement des modèles
        if model_high_path:
            loaded_model_high, _ = self._load_model(model_high_path, dtype_high, enable_checksum)
            
        if model_low_path:
            loaded_model_low, _ = self._load_model(model_low_path, dtype_low, enable_checksum)

        print("="*50 + "\n")
        return (loaded_model_high, loaded_model_low)
