# 🤖 XT-404 Skynet : Wan 2.2 Sentinel Suite
### Cyberdyne Systems Corp. | Series T-800 | Model 101

<p align="center">
  <img src="https://img.shields.io/badge/Version-v3.5_Omega-red?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Wan_2.2-blue?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Engine-Tesseract_V6-purple?style=for-the-badge" alt="Engine">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License">
</p>

> *"The future is not set. There is no fate but what we make for ourselves."*

---

## ⚠️ CRITICAL SYSTEM DEPENDENCY / DÉPENDANCE CRITIQUE

> [!CAUTION]
> **INFILTRATION PROTOCOL (GGUF):**
> To utilize GGUF Quantized Models with this suite, the **ComfyUI-GGUF** engine is **REQUIRED**.
>
> 📥 **Download Engine:** `city96/ComfyUI-GGUF`
>
> *Without this engine, the Cyberdyne Model Hub will operate in Safetensors-only mode.*

---

## 🌍 NEURAL NET NAVIGATION / NAVIGATION DU RÉSEAU

### 🇺🇸 [ENGLISH DOCUMENTATION](#-english-documentation)
1. [Latest Intel (Omega Changelog)](#-latest-intel-omega-v35-changelog)
2. [Phase 1: Infiltration (Loaders)](#%EF%B8%8F-phase-1-infiltration-loaders)
3. [Phase 2: Neural Net Core (Samplers)](#-phase-2-neural-net-core-samplers-xt-404)
4. [Phase 3: Hardware Optimization (The Omega Engine)](#-phase-3-hardware-optimization-omega-engine)
5. [Phase 4: Post-Processing & Tools](#%EF%B8%8F-phase-4-post-processing--tools)

### 🇫🇷 [DOCUMENTATION FRANÇAISE](#-documentation-française)
1. [Dernières Infos (Mise à jour Omega)](#-dernières-infos-omega-v35-mise-à-jour)
2. [Phase 1 : Infiltration (Chargement)](#%EF%B8%8F-phase-1--infiltration-chargement)
3. [Phase 2 : Cœur Neuronal (Samplers)](#-phase-2--cœur-neuronal-samplers-xt-404)
4. [Phase 3 : Optimisation Matérielle (Moteur Omega)](#-phase-3--optimisation-matérielle-moteur-omega)
5. [Phase 4 : Post-Production & Outils](#%EF%B8%8F-phase-4--post-production--outils)

---

# 🇺🇸 ENGLISH DOCUMENTATION

## 📡 Latest Intel (Omega v3.5 Changelog)

XT-404 Skynet is an elite engineering suite for ComfyUI, specifically architected for the Wan 2.2 video generation model. The **v3.5 Omega** update introduces "Self-Healing" capabilities and Asynchronous pipelines.

### 🆕 System Status Update (v3.5 Omega):

*   **TeaCache "Chronos Sentinel" (V5):**
    *   **Step-Counter Hard Lock:** Forces the calculation of the first 2 steps regardless of similarity. Essential for Turbo workflows (6 steps) to prevent image collapse.
    *   **Dual-Flow Engine:** Separates Positive and Negative prompt caching to prevent CFG collisions.
    *   **Quantum Safe (FP8):** Automatically detects FP8 quantization conflicts and disables `Autocast` on the fly to prevent crashes.
*   **VRAM Guard "Tesseract Engine" (V6):**
    *   **Async Transfer:** Decodes the next frame chunk while the previous one is being copied to RAM. Eliminates UI lag.
    *   **Zero-Lag GC:** Removed aggressive garbage collection from critical loops.
*   **Samplers (Passive Mode):** The "Vector Amplification" has been decommissioned. The Sentinel now operates in Passive Monitoring Mode, fixing "burn" issues in Refined workflows.

---

## 🛡️ Phase 1: Infiltration (Loaders)

### 🤖 Cyberdyne Model Hub
**Class:** `CyberdyneModelHub`

A unified, intelligent loader that bridges the gap between Analog (Safetensors) and Quantized (GGUF) architectures. It specifically handles the Wan 2.2 Dual-Model requirement (High Context + Low Context).

| Parameter | Description |
| :--- | :--- |
| `model_high_name` | The main UNet model. Supports `.safetensors` AND `.gguf`. |
| `dtype_high` | Precision override (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | The secondary UNet model (Wan 2.2 requirement). |
| `enable_checksum` | Performs a SHA256 integrity scan (Security Protocol). |
| `offload_inactive` | **"Skynet Protocol":** Aggressively purges VRAM of unused models before loading new ones. |

---

## 🧠 Phase 2: Neural Net Core (Samplers XT-404)

The "Sentinel" engine powers three specialized sampling nodes designed for chained workflows.

### 🔴 XT-404 Skynet 1 (Master)
**The Commander node.** Initializes generation and defines the global noise schedule.
*   **Outputs:** Latent, Denoised Latent, Options (for chaining), Seed.

| Parameter | Description |
| :--- | :--- |
| `sampler_name` | Combo selection (e.g., `linear/euler`, `beta/dpmpp_2m`). |
| `cfg` | Guidance Scale. Monitored by Sentinel. |
| `bongmath` | Texture Engine. `True` = Film/Analog look. `False` = Digital/Smooth. |
| `sampler_mode` | Standard or Resample (injects fresh noise). |

### 🟡 XT-404 Skynet 2 (Chain)
**The Relay node.** Designed for split-sampling (e.g., first 50% on Master, next 30% on Chain).
*   **Logic:** Hides the Seed widget (uses internal deterministic inheritance).
*   **VRAM:** Dynamic unloading based on model type.

### 🟢 XT-404 Skynet 3 (Refiner)
**The Terminator node.** Finalizes the image structure.
*   **Configuration:** `steps_to_run` defaults to `-1` (finish the schedule).

---

## ⚡ Phase 3: Hardware Optimization (Omega Engine)

### 🚀 Wan Hardware Accelerator
**Class:** `Wan_Hardware_Accelerator`
Enables low-level PyTorch optimizations (**TF32**) for NVIDIA Ampere+ GPUs.
*   **TF32:** Increases speed by ~20% on compatible GPUs with negligible precision loss.

### 🧩 Wan Hybrid VRAM Guard (Omega V6)
**Class:** `Wan_Hybrid_VRAM_Guard`
**The Tesseract Engine.** Replaces standard VAE Decode for Wan 2.2.
*   **Async Stream:** Pipeline architecture (Decode -> Transfer -> Save) happens simultaneously.
*   **Pin Memory:** Uses pinned CPU RAM for DMA transfers (Direct Memory Access).
*   **Tiling:** Forces spatial tiling (512px chunks) to fit 8K video in 12GB VRAM.

### 🍵 Wan TeaCache (Omega V5)
**Class:** `Wan_TeaCache_Patch`
**The Chronos Sentinel.** Caches U-Net outputs to speed up generation (1.5x - 2x).
*   **Quantum Safe:** Set `force_autocast` to `False` for GGUF/FP8 models to avoid "ScalarType" errors. The node will auto-correct if you forget.
*   `rel_l1_threshold`:
    *   **0.05 - 0.1**: Safe zone for high quality.
    *   **0.02**: Required for Turbo/Lightning (6 steps) workflows.

---

## 🛠️ Phase 4: Post-Processing & Tools

### 💾 Wan Video Compressor (H.265)
**Class:** `Wan_Video_Compressor`
Encodes output to H.265 10-bit.
*   **Modes:** Web/Discord (<5MB target), Master (High Fidelity), Archival.

### 🧹 Wan Cycle Terminator
**Class:** `Wan_Cycle_Terminator`
Surgical memory cleaning using Windows API `EmptyWorkingSet`. Use only when switching heavy workflows.

### 📐 Resolution Savant
**Class:** `Wan_Resolution_Savant`
Resizes images ensuring dimensions are strictly divisible by 16 (Wan Requirement).
*   **Modes:** `lanczos` (CPU/High Quality) or `bicubic/area` (GPU/Fast).

---
---

# 🇫🇷 DOCUMENTATION FRANÇAISE

## 📡 Dernières Infos (Omega v3.5 Mise à jour)

XT-404 Skynet est une suite d'ingénierie d'élite pour ComfyUI. La mise à jour **v3.5 Omega** introduit des capacités d'auto-réparation et des pipelines asynchrones.

### 🆕 Mise à jour État Système (v3.5 Omega) :

*   **TeaCache "Chronos Sentinel" (V5) :**
    *   **Verrouillage Physique (Hard Lock) :** Force le calcul des 2 premières étapes quoi qu'il arrive. Vital pour les workflows Turbo (6 steps).
    *   **Moteur Double Flux (Dual-Flow) :** Sépare le cache du prompt Positif et Négatif pour éviter les collisions de CFG.
    *   **Sécurité Quantique (FP8) :** Détecte automatiquement les conflits de quantification FP8 et désactive `Autocast` à la volée pour éviter les crashs.
*   **VRAM Guard "Moteur Tesseract" (V6) :**
    *   **Transfert Asynchrone :** Décode le morceau (chunk) suivant pendant que le précédent est copié en RAM. Élimine le lag de l'interface.
    *   **Zero-Lag GC :** Suppression du nettoyage mémoire agressif dans les boucles critiques.
*   **Samplers (Mode Passif) :** L'amplification vectorielle a été désactivée pour éviter la sur-saturation.

---

## 🛡️ Phase 1 : Infiltration (Chargement)

### 🤖 Cyberdyne Model Hub
**Classe :** `CyberdyneModelHub`

Un chargeur unifié qui gère l'exigence Wan 2.2 Dual-Model (High + Low Context) et supporte nativement les fichiers GGUF.

| Paramètre | Description |
| :--- | :--- |
| `model_high_name` | Modèle principal. Supporte `.safetensors` ET `.gguf`. |
| `dtype_high` | Forçage précision (`fp16`, `bf16`, `fp8_e4m3fn`, etc.). |
| `model_low_name` | Modèle secondaire (Requis par Wan 2.2). |
| `enable_checksum` | Scan d'intégrité SHA256. |
| `offload_inactive` | **"Protocole Skynet" :** Purge la VRAM avant chargement. |

---

## 🧠 Phase 2 : Cœur Neuronal (Samplers XT-404)

### 🔴 XT-404 Skynet 1 (Master)
**Le Commandant.** Initialise la génération.
*   **Bongmath :** Moteur de Texture. `True` = Grain Film. `False` = Lisse.

### 🟡 XT-404 Skynet 2 (Chain)
**Le Relais.** Conçu pour l'échantillonnage fractionné. Masque le Seed pour héritage déterministe.

### 🟢 XT-404 Skynet 3 (Refiner)
**Le Terminator.** Finalise les détails haute fréquence.

---

## ⚡ Phase 3 : Optimisation Matérielle (Moteur Omega)

### 🚀 Wan Hardware Accelerator
**Classe :** `Wan_Hardware_Accelerator`
Active **TF32** sur les GPU NVIDIA Ampere+. Gain de vitesse ~20%.

### 🧩 Wan Hybrid VRAM Guard (Omega V6)
**Classe :** `Wan_Hybrid_VRAM_Guard`
**Le Moteur Tesseract.** Remplace le Decode VAE standard.
*   **Flux Asynchrone :** Architecture Pipeline (Décodage -> Transfert -> Sauvegarde) simultanée.
*   **Pin Memory :** Utilise la RAM CPU verrouillée pour des transferts DMA ultra-rapides.
*   **Tuilage :** Découpe l'image en blocs de 512px.

### 🍵 Wan TeaCache (Omega V5)
**Classe :** `Wan_TeaCache_Patch`
**La Sentinelle Chronos.** Cache les sorties U-Net pour accélérer la génération.
*   **Sécurité Quantique :** Mettre `force_autocast` sur `False` pour les modèles GGUF/FP8. Le nœud se corrige tout seul si vous oubliez.
*   `rel_l1_threshold` :
    *   **0.05 - 0.1** : Qualité standard.
    *   **0.02** : Requis pour les workflows Turbo (6 steps).

---

## 🛠️ Phase 4 : Post-Production & Outils

### 💾 Wan Video Compressor
Encode la sortie en H.265 10-bits (Web, Master, Archival).

### 🧹 Wan Cycle Terminator
Nettoyage chirurgical de la mémoire via API Windows.

### 📐 Resolution Savant
Redimensionne les images pour qu'elles soient divisibles par 16.
*   **Modes :** `lanczos` (CPU/Qualité) ou `bicubic/area` (GPU/Vitesse).

---

<p align="center">
  <i>Architected by Cyberdyne Systems. No fate but what we make.</i>
</p>
