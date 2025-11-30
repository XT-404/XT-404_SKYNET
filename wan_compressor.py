import os
import subprocess
import sys
import time
import shutil
from typing import List, Union

# --- AUTO-INSTALL DEPENDENCY CHECK ---
print(">> [Wan Architect] Initializing Compression Engine (Omega Optimized)...")
def check_dependencies():
    try:
        import imageio_ffmpeg
    except ImportError:
        print(">> [Wan Dependency] Installing optimized codecs...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
        except Exception as e:
            print(f"!! [Wan Critical] Install failed: {e}")
check_dependencies()

# --- UTILS ---
class AnyType(str):
    def __ne__(self, __value: object) -> bool: return False

class Wan_Video_Compressor:
    """
    OMEGA EDITION (Fix High-End CPU Crash):
    - Thread-Safe : Limitation intelligente (Min(Cores-2, 16)) pour respecter la limite x265.
    - OS Friendly : Exécution en 'Low Priority' pour ne pas figer la souris.
    - Smart Encoding : H.265 10-bit 'Medium'.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": (AnyType("*"), {"forceInput": True}), 
                "mode": ([
                    "Web/Discord (Balanced - CRF 26)", 
                    "Master (High Fidelity - CRF 22)", 
                    "Archival (Lossless - CRF 18)"
                ], {"default": "Web/Discord (Balanced - CRF 26)"}),
                "remove_original": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_grain_texture": ("BOOLEAN", {"default": True, "tooltip": "Préserve le grain (Psy-RD) sans saturation CPU/Bitrate."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compressed_video_path",)
    FUNCTION = "compress_video"
    CATEGORY = "ComfyWan_Architect/PostProcessing"
    OUTPUT_NODE = True

    def _get_ffmpeg(self):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe): return exe
        except: pass
        
        if shutil.which("ffmpeg"): return "ffmpeg"
        
        local = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
        if os.path.exists(local): return local
            
        raise RuntimeError("CRITICAL: FFmpeg binary not found. Please install imageio-ffmpeg.")

    def _calculate_threads(self):
        # OMEGA FIX v2: Gestion des CPU High-End (Threadripper/i9/Ryzen 9)
        count = os.cpu_count()
        if count:
            # 1. On laisse respirer l'OS (-2 coeurs)
            val = count - 2
            # 2. CAP HARDWARE X265 : La lib refuse souvent > 16 threads (Crash -1094995529)
            # De toute façon, au-delà de 12-16, le gain est nul.
            val = min(val, 16)
            return str(max(1, val))
        return "1"

    def _recursive_find_video(self, data):
        valid_exts = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
        found = []
        if isinstance(data, str):
            if data.lower().endswith(valid_exts) and os.path.exists(data):
                found.append(data)
        elif isinstance(data, (list, tuple)):
            for item in data:
                found.extend(self._recursive_find_video(item))
        elif isinstance(data, dict):
            for value in data.values():
                found.extend(self._recursive_find_video(value))
        return found

    def compress_video(self, video_path, mode, remove_original, keep_grain_texture):
        target_files = self._recursive_find_video(video_path)
        if not target_files: 
            return (video_path,)

        ffmpeg_exe = self._get_ffmpeg()
        output_paths = []
        
        # --- OMEGA CONFIG ---
        codec = "libx265"
        preset = "medium" 
        safe_threads = self._calculate_threads()
        
        x265_params = ["aq-mode=3", "log-level=error"]

        if "Web/Discord" in mode:
            crf = "26" 
            suffix = "_optm"
            x265_params.append("vbv-maxrate=6000:vbv-bufsize=12000")
        elif "Master" in mode:
            crf = "22"
            suffix = "_mstr"
        else:
            crf = "18"
            suffix = "_arch"

        if keep_grain_texture:
            x265_params.append("psy-rd=2.0:psy-rdoq=1.0")
        else:
            x265_params.append("psy-rd=1.0")

        x265_params_str = ":".join(x265_params)

        for input_file in target_files:
            dir_name = os.path.dirname(input_file)
            file_name = os.path.basename(input_file)
            name_no_ext, ext = os.path.splitext(file_name)
            
            if suffix in name_no_ext:
                output_paths.append(input_file)
                continue

            output_file = os.path.join(dir_name, f"{name_no_ext}{suffix}.mp4")
            
            print(f">> [Wan Omega] Processing: {file_name}")
            print(f"   Settings: Preset '{preset}' | CRF {crf} | Threads {safe_threads} (Capped for x265 Stability)")

            cmd = [
                ffmpeg_exe, "-y", "-v", "error", "-stats",
                "-i", input_file,
                "-threads", safe_threads,  # FIX: Capped à 16 max
                "-c:v", codec,
                "-crf", crf,
                "-preset", preset,
                "-x265-params", x265_params_str,
                "-pix_fmt", "yuv420p10le",
                "-movflags", "+faststart", 
                "-c:a", "copy",
                output_file
            ]

            try:
                # --- PROCESS PRIORITY ---
                startupinfo = None
                creationflags = 0
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    creationflags = 0x00004000 # BELOW_NORMAL_PRIORITY_CLASS
                
                start_t = time.time()
                subprocess.run(
                    cmd, 
                    check=True, 
                    startupinfo=startupinfo, 
                    creationflags=creationflags
                )
                end_t = time.time()

                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    orig_s = os.path.getsize(input_file) / (1024*1024)
                    new_s = os.path.getsize(output_file) / (1024*1024)
                    reduction = (1 - (new_s / orig_s)) * 100
                    
                    print(f"   [Done] {orig_s:.2f}MB -> {new_s:.2f}MB (-{reduction:.1f}%) in {end_t-start_t:.1f}s")
                    output_paths.append(output_file)
                    
                    if remove_original:
                        try: os.remove(input_file)
                        except: pass
                else:
                    output_paths.append(input_file)

            except Exception as e:
                print(f"!! [Wan Error] Encoding failed: {e}")
                output_paths.append(input_file)

        if len(output_paths) == 1: 
            return (output_paths[0],)
        return (output_paths,)

# --- ESSENTIAL MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "Wan_Video_Compressor": Wan_Video_Compressor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan_Video_Compressor": "Wan 2.2 Video Compressor (Omega)"
}
