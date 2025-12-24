# XT-404 SKYNET SUITE: Wan 2.2 Integration Protocol
### Cyberdyne Systems Corp. | Series T-800 | Model 101

![Version](https://img.shields.io/badge/Version-3.7_Gold_Master-cyan?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-BATTLE_TESTED-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-ComfyUI-blue?style=for-the-badge)

## 💀 System Overview

The **XT-404 Skynet Suite** is a highly advanced, custom node collection for **ComfyUI**, specifically engineered to optimize, accelerate, and stabilize the **Wan 2.1 / 2.2 Video Diffusion Models**.

Unlike standard implementations, this suite introduces "Cyberdyne" architecture: a set of overrides, caching mechanisms, and mathematical corrections (Zero-Point injection, Inverse Structural Repulsion, Contextual TF32 Switching) to solve common issues like image burning, OOM (Out of Memory) crashes, and temporal flickering.

---
## ⚠️ Requirements

*   **ComfyUI:** Latest version recommended.
*   **Wan 2.2 Models:** Ensure you have the VAE, CLIP, and UNet/Transformer models.
*   **Python:** 3.10+.
*   **FFmpeg:** Required for the Compressor node (usually installed via `imageio-ffmpeg`).
## ⚠️ SYSTEM DEPENDENCY

> [!CAUTION]
> **INFILTRATION PROTOCOL (GGUF):**
> To utilize GGUF Quantized Models with the **Cyberdyne Model Hub**, the **ComfyUI-GGUF** engine is **REQUIRED**.
>
> 📥 **Download Engine:** `city96/ComfyUI-GGUF`
>
> *Without this engine, the Cyberdyne Model Hub will operate in Safetensors-only mode.*
---

## 📂 Core Architecture & Module Breakdown

### 1. Neural Net Core (Sampling & Control)

#### `XT404_Skynet_Nodes.py`
**The Brain.** This module replaces the standard KSampler with a synchronized multi-stage sampling system tailored for video consistency.
*   **XT404_Skynet_1 (Master):** The primary clock generator. Calculates the "Flow Matching" sigmas using a custom shift value (default 5.0). It injects "Zero-Point" noise to eliminate static snow in the initial latent.
*   **XT404_Skynet_2 (Chain) & 3 (Refiner):** These nodes lock onto the Master's timeline. They allow for seamless hand-offs between steps (e.g., Step 0-15 on Master, 15-20 on Refiner) without breaking the temporal coherence.
*   **Key Tech:** *Bongmath* (Precision float math on CPU), *Zero-Point Injection*.

#### `wan_tx_node.py`
**The Polymetric Interpolator (T-X).** A specialized interpolator for frame generation.
*   **Dual Phase Latent Bridging:** Allows smooth transitions between a `start_image` and an `end_image`.
*   **Inverse Structural Repulsion (ISR):** A motion engine that isolates high-frequency structure from color data to boost motion amplitude without ghosting.
*   **Wan-Specific Masking:** Automatically handles the complex 4-frame block masking required by the Wan architecture.

---

### 2. Mimetic Rendering (Generation Engines)

#### `nodes_wan_ultra.py`
**Wan Image To Video Ultra (Hybrid V19).** The heavy-lifting generator node.
*   **Smart VRAM Scanner:** Automatically detects GPU memory (e.g., >22GB) to determine the optimal tile strategy (1280px vs 1024px vs 512px).
*   **Hybrid Motion Fix:** Reverts to a "Classic" algorithm (Linear + Mean Centering) for motion amplification to prevent the black/saturated artifacts caused by soft limiters in previous versions.
*   **Tiled Encoding:** Supports large resolution encoding by breaking images into tiles with overlap.

#### `wan_fast.py`
**Wan Image To Video Fidelity.** A streamlined, high-fidelity version of the generator.
*   **Full Context Encoding:** Unlike the Ultra version which may tile, this node forces full-frame context awareness for maximum coherence.
*   **FP32 Enforcement:** Forces 32-bit floating-point precision during the upscale and VAE encode stages to prevent color banding.
*   **Aggressive Cleanup:** Immediately purges source tensors from VRAM after encoding.

---

### 3. System Optimization & Protection (The T-3000 Series)

#### `wan_accelerator.py`
**Hardware Accelerator (Anti-Burn V4).** Solves the "Fried/Burnt Image" issue common with Wan models on consumer GPUs.
*   **Contextual Precision Switching:** Globally enables TF32 (TensorFloat-32) for speed, *but* surgically installs "interrupters" on GroupNorm and LayerNorm layers. These sensitive layers are forced to run in native FP32, providing the speed of TF32 with the quality of FP32.
*   **Wan_Attention_Slicer:** Manages SDPA (Scaled Dot Product Attention) memory usage.

#### `wan_genisys.py` (aka `wan_optimizer.py`)
**Cyberdyne Genisys [Nano-Repair / Omniscient].** An advanced caching and stabilization system.
*   **Drift Detection:** Monitors the latent difference between steps. If the change (Drift) is below a threshold (`security_level`), it skips the UNet calculation and reuses the previous output (Cache Hit).
*   **Nano-Repair:** Detects NaNs (Not a Number) or infinite values in the tensor stream and clamps them to a safe range (+/- 10.0), preventing black frames.
*   **HUD:** Prints a real-time tactical dashboard in the console showing signal integrity and load.

#### `wan_cleanup.py`
**Cycle Terminator.** A memory management node.
*   **Surgical Purge:** Uses the Windows PSAPI `EmptyWorkingSet` (or `malloc_trim` on Linux) to release RAM that Python/PyTorch has freed but the OS hasn't reclaimed.
*   **Skynet Quotes:** Prints randomized T-800 quotes upon execution.

---

### 4. Infiltration & IO (Loaders)

#### `cyberdyne_model_hub.py`
**Universal Model Loader.**
*   **Recursive Scanning:** Finds models (Safetensors, CKPT, GGUF) in subdirectories automatically.
*   **Integrity Check:** Performs SHA256 checksums on load to ensure model validity.
*   **GGUF Support:** Automatically delegates GGUF loading to `UnetLoaderGGUF` if available.

#### `__init__.py`
**Boot Sequence.**
*   Initializes the suite, performs dependency checks (scikit-image, imageio), and prints the ASCII HUD.
*   Maps all nodes to ComfyUI.

---

### 5. Automation & Tools

#### `auto_wan_node.py`
**Auto Wan 2.2 Optimizer.**
*   **Modulo 16 Safety:** Automatically resizes images so dimensions are divisible by 16 (required by Wan).
*   **OOM Protection:** Downscales images if they exceed 1024px on the longest side.
*   **Min Size Fix:** Ensures no dimension is smaller than 512px.

#### `auto_half_node.py`
**Auto Half Size.**
*   Simple utility to perform a high-quality (Bicubic + Antialias) 50% downscale.

#### `wan_i2v_tools.py`
**Vision & Resolution Tools.**
*   **Vision OneShot Cache (Omega):** Hashes input images to cache CLIP Vision outputs, preventing redundant encoding.
*   **Resolution Savant:** Provides "Lanczos" (CPU-based) resampling for ultimate quality, or FP32 GPU resampling for speed.

#### `wan_text_encoder.py`
**Text OneShot Cache.**
*   Pins text embeddings in RAM (DMA) for instant transfer to GPU, optimizing prompt encoding speed.

---

### 6. Post-Processing & Mastering

#### `wan_chroma_mimic.py`
**Chroma Mimic (Turbo OLED).** A GPU-accelerated mastering node.
*   **LAB Transfer:** Applies the color palette of a reference image to the video using the LAB color space.
*   **OLED Dynamics:** Applies an S-Curve contrast boost for deep blacks.
*   **Surface Blur:** Smooths skin and metal textures while preserving edge sharpness.

#### `wan_compressor.py`
**Video Compressor (Omega).**
*   **Thread Safety:** Limits FFmpeg threads (max 16) to prevent specific CPU crashes (Ryzen 9 / Threadripper issue with x265).
*   **H.265 10-bit:** Encodes in high-efficiency format suitable for web or archival.

---

## 🚀 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/XT404-Skynet-Suite.git
    ```
3.  Install dependencies (The suite attempts to auto-install, but manual is recommended):
    ```bash
    pip install imageio-ffmpeg scikit-image
    ```
4.  **Restart ComfyUI.** Watch the console for the "CYBERDYNE SYSTEMS" boot log.

---
*“The future is not set. There is no fate but what we make for ourselves.”*
