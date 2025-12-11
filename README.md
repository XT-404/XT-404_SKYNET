# 🤖 XT-404 Skynet : Wan 2.2 Sentinel Suite
### Cyberdyne Systems Corp. | Series T-800 | Model 101

<p align="center">
  <img src="https://img.shields.io/badge/Version-v4.0_Omega_MagCache-red?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Wan_2.2-blue?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Engine-T_1000_Sentinel-purple?style=for-the-badge" alt="Engine">
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
1. [Latest Intel (MagCache & T-1000)](#-latest-intel-omega-v40-magcache--t-1000)
2. [Phase 1: Infiltration (Loaders)](#%EF%B8%8F-phase-1-infiltration-loaders)
3. [Phase 2: Neural Net Core (Samplers)](#-phase-2-neural-net-core-samplers-xt-404)
4. [Phase 3: Hardware Optimization (MagCache Omega)](#-phase-3-hardware-optimization-omega-engine)
5. [Phase 4: Post-Processing & Tools](#%EF%B8%8F-phase-4-post-processing--tools)
6. [Phase 6: Mimetic Generation (Fidelity & Ultra)](#-phase-6-mimetic-generation-fidelity--ultra)

### 🇫🇷 [DOCUMENTATION FRANÇAISE](#-documentation-française)
1. [Dernières Infos (MagCache & T-1000)](#-dernières-infos-omega-v40-magcache--t-1000)
2. [Phase 1 : Infiltration (Chargement)](#%EF%B8%8F-phase-1--infiltration-chargement)
3. [Phase 2 : Cœur Neuronal (Samplers)](#-phase-2--cœur-neuronal-samplers-xt-404)
4. [Phase 3 : Optimisation Matérielle (MagCache Omega)](#-phase-3--optimisation-matérielle-moteur-omega)
5. [Phase 4 : Post-Production & Outils](#%EF%B8%8F-phase-4--post-production--outils)
6. [Phase 6 : Génération Mimétique (Fidelity & Ultra)](#-phase-6--génération-mimétique-fidelity--ultra)

---

# 🇺🇸 ENGLISH DOCUMENTATION

## 📡 Latest Intel (Omega v4.0: MagCache & T-1000)

XT-404 Skynet is an elite engineering suite for ComfyUI, specifically architected for the Wan 2.2 video generation model. The **v4.0 Omega** update replaces the legacy TeaCache with the revolutionary **MagCache** and introduces the **T-1000 Sentinel** telemetry system.

### 🆕 System Status Update (v4.0 Omega):

*   **MagCache "Omega Edition" (Replaces TeaCache):**
    *   **Accumulated Error Logic:** Unlike TeaCache (instant delta), MagCache accumulates signal drift over time. It only triggers a recalculation when the total drift exceeds the threshold. This guarantees prompt fidelity over long sequences.
    *   **Quantum Safe (FP8/BF16):** Includes a specific fix for "QuantizedTensor" crashes. It casts tensors to FP32 *only* for metric calculation, ensuring 100% stability with quantized models.
    *   **Dual-Flow Engine:** Completely isolates Positive and Negative prompt caching via memory pointer analysis (`data_ptr`). Prevents signal cross-contamination.
*   **T-1000 Sentinel (Active Telemetry):**
    *   **Real-Time Console HUD:** Displays Step, Flow ID, Signal Drift, and Fidelity % in the ComfyUI console.
    *   **Turbo Hard Lock:** Automatically detects aggressive samplers (Turbo/Lightning 6-steps) and **forces** calculation for the first few steps to prevent image collapse.
*   **Samplers (XT-404):** Optimized for "Shift" parameter handling (default 5.0) for Wan 2.2.

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
| `shift_val` | **Critical for Wan 2.2**. Default **5.0**. Controls the noise schedule curve. |
| `bongmath` | Texture Engine. `True` = Film/Analog look. `False` = Digital/Smooth. |
| `sampler_mode` | Standard or Resample (injects fresh noise). |

### 🟡 XT-404 Skynet 2 (Chain)
**The Relay node.** Designed for split-sampling.
*   **Logic:** Hides the Seed widget (uses internal deterministic inheritance).

### 🟢 XT-404 Skynet 3 (Refiner)
**The Terminator node.** Finalizes the image structure.

---

## ⚡ Phase 3: Hardware Optimization (Omega Engine)

### 🔮 Wan MagCache (T-1000 Sentinel)
**Class:** `Wan_MagCache_Patch`
**The Evolution of Caching.** Replaces TeaCache. Analyzes signal magnitude to skip redundant steps.

| Parameter | Description |
| :--- | :--- |
| `enable_mag_cache` | Toggle the system on/off. |
| `mag_threshold` | **0.020** (Default). The accumulated error limit. Higher = Faster/Lower Quality. Lower = Slower/Higher Quality. |
| `start_step_percent`| **0.3** (Default). Forces calculation for the first 30% of steps. Critical for structure. |
| `verbose_t1000` | **TRUE**. Activates the T-1000 HUD in the console to monitor Signal Fidelity %. |

### 🚀 Wan Hardware Accelerator
**Class:** `Wan_Hardware_Accelerator`
Enables low-level PyTorch optimizations (**TF32**) for NVIDIA Ampere+ GPUs.

### 🧩 Wan Hybrid VRAM Guard (Native Pass)
**Class:** `Wan_Hybrid_VRAM_Guard`
Maintained for workflow compatibility. Uses ComfyUI's native optimized decoder.

---

## 🛠️ Phase 4: Post-Processing & Tools

### 💾 Wan Video Compressor (H.265)
**Class:** `Wan_Video_Compressor`
Encodes output to H.265 10-bit with CPU thread management (prevents system lag).

### 🧹 Wan Cycle Terminator
**Class:** `Wan_Cycle_Terminator`
Surgical memory cleaning using Windows API `EmptyWorkingSet`. Use only when switching heavy workflows.

### 📐 Resolution Savant (FP32)
**Class:** `Wan_Resolution_Savant`
Resizes images ensuring dimensions are strictly divisible by 16. Uses **FP32 interpolation** to prevent color banding.

---

## 🎭 Phase 6: Mimetic Generation (Fidelity & Ultra)

**Class:** `WanImageToVideoUltra` | `WanImageToVideoFidelity`

The **"Ultra"** engine is a complete re-engineering of the standard I2V process, focusing on absolute FP32 precision and dynamic control.

### 🌟 Key Features (Ultra Node)
*   **FP32 Forced Pipeline:** All math runs in 32-bit floating point to eliminate color banding.
*   **Bicubic AA:** High-end upscaling with anti-aliasing (better than bilinear).
*   **Detail Boost (Paroxysm):** A GPU-sharpening matrix applied *before* encoding to counteract VAE blur.
*   **Motion Amplification:** A mathematical dynamic booster that solves "static video" issues.
*   **Reference Injection:** Forces the model to "remember" the source identity (Identity Lock).
*   **Smart Duration:** Dropdown menu for exact time (5s, 10s...) and frame calculation.

| Parameter | Description |
| :--- | :--- |
| `video_duration` | Dropdown (e.g., "5s (114 frames)"). Auto-calculates correct frame count. |
| `detail_boost` | Sharpening factor. **0.5** is recommended for natural HD. |
| `motion_amp` | Dynamic booster. **1.0** = Normal. **1.15** = Forced movement. |
| `force_ref` | **True** = Injects source image as a hard reference (High Fidelity). |

---
---

# 🇫🇷 DOCUMENTATION FRANÇAISE

## 📡 Dernières Infos (Omega v4.0 : MagCache & T-1000)

XT-404 Skynet est une suite d'ingénierie d'élite pour ComfyUI. La mise à jour **v4.0 Omega** remplace l'ancien TeaCache par le révolutionnaire **MagCache** et déploie le système de télémétrie **T-1000 Sentinel**.

### 🆕 Mise à jour État Système (v4.0 Omega) :

*   **MagCache "Omega Edition" (Remplace TeaCache) :**
    *   **Logique d'Erreur Accumulée :** Contrairement au TeaCache (différence instantanée), le MagCache accumule la dérive du signal dans le temps. Il ne recalcule que lorsque la dérive totale dépasse le seuil. Cela garantit une fidélité parfaite au prompt sur les longues séquences.
    *   **Sécurité Quantique (FP8/BF16) :** Intègre un correctif spécifique pour les crashs "QuantizedTensor". Il convertit les tenseurs en FP32 *uniquement* pour le calcul métrique, assurant une stabilité à 100% avec les modèles GGUF/Quantized.
    *   **Moteur Double Flux (Dual-Flow) :** Isole totalement le cache des prompts Positifs et Négatifs via une analyse des pointeurs mémoire (`data_ptr`). Empêche la contamination du signal.
*   **T-1000 Sentinel (Télémétrie Active) :**
    *   **HUD Console Temps Réel :** Affiche l'étape, l'ID du flux, la dérive (Drift) et le % de Fidélité dans la console ComfyUI.
    *   **Verrouillage Turbo (Hard Lock) :** Détecte automatiquement les samplers agressifs (Turbo/Lightning 6-steps) et **force** le calcul des premières étapes pour éviter l'effondrement de l'image (status `[LOCKED]`).

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
*   **Shift Val :** **5.0** (Défaut). Crucial pour Wan 2.2. Contrôle la courbe de bruit.
*   **Bongmath :** Moteur de Texture. `True` = Grain Film. `False` = Lisse.

### 🟡 XT-404 Skynet 2 (Chain)
**Le Relais.** Conçu pour l'échantillonnage fractionné. Masque le Seed pour héritage déterministe.

### 🟢 XT-404 Skynet 3 (Refiner)
**Le Terminator.** Finalise les détails haute fréquence.

---

## ⚡ Phase 3 : Optimisation Matérielle (MagCache Omega)

### 🔮 Wan MagCache (T-1000 Sentinel)
**Classe :** `Wan_MagCache_Patch`
**L'Évolution du Cache.** Remplace le TeaCache. Analyse la magnitude du signal pour sauter les étapes redondantes.

| Paramètre | Description |
| :--- | :--- |
| `enable_mag_cache` | Active ou désactive le système. |
| `mag_threshold` | **0.020** (Défaut). Seuil d'erreur accumulée. Plus haut = Plus rapide. Plus bas = Meilleure qualité. |
| `start_step_percent`| **0.3** (Défaut). Force le calcul sur les premiers 30% des étapes. Vital pour la structure. |
| `verbose_t1000` | **TRUE**. Active le HUD T-1000 dans la console pour surveiller le % de Fidélité. |

### 🚀 Wan Hardware Accelerator
**Classe :** `Wan_Hardware_Accelerator`
Active **TF32** sur les GPU NVIDIA Ampere+. Gain de vitesse ~20%.

### 🧩 Wan Hybrid VRAM Guard (Native Pass)
**Classe :** `Wan_Hybrid_VRAM_Guard`
Maintenu pour la compatibilité des workflows. Utilise le décodeur natif optimisé de ComfyUI.

---

## 🛠️ Phase 4 : Post-Production & Outils

### 💾 Wan Video Compressor
Encode la sortie en H.265 10-bits avec gestion intelligente des cœurs CPU.

### 🧹 Wan Cycle Terminator
Nettoyage chirurgical de la mémoire via API Windows.

### 📐 Resolution Savant (FP32)
Redimensionne les images pour qu'elles soient divisibles par 16. Utilise l'interpolation **FP32** pour éviter le banding des couleurs.

---

## 🎭 Phase 6 : Génération Mimétique (Fidelity & Ultra)

**Classe :** `WanImageToVideoUltra` | `WanImageToVideoFidelity`

Le moteur **"Ultra"** est une ré-ingénierie complète du processus I2V standard, axée sur la précision FP32 absolue et le contrôle dynamique.

### 🌟 Fonctionnalités Clés (Nœud Ultra)
*   **Pipeline FP32 Forcé :** Tous les calculs mathématiques se font en 32 bits pour éliminer les bandes de couleurs.
*   **Bicubic AA :** Upscaling haut de gamme avec anti-aliasing (meilleur que le bilinéaire).
*   **Detail Boost (Paroxysme) :** Une matrice de netteté GPU appliquée *avant* l'encodage pour contrer le flou du VAE.
*   **Amplification de Mouvement :** Un booster dynamique mathématique qui résout les problèmes de "vidéo statique".
*   **Injection de Référence :** Force le modèle à "se souvenir" de l'identité de la source (Identity Lock).
*   **Durée Intelligente :** Menu déroulant pour le temps exact (5s, 10s...) et le calcul des frames.

| Paramètre | Description |
| :--- | :--- |
| `video_duration` | Menu (ex: "5s (114 frames)"). Calcule auto le bon nombre de frames. |
| `detail_boost` | Facteur de netteté. **0.5** recommandé pour un HD naturel. |
| `motion_amp` | Booster dynamique. **1.0** = Normal. **1.15** = Mouvement forcé. |
| `force_ref` | **True** = Injecte l'image source comme référence dure (Haute Fidélité). |

---

<p align="center">
  <i>Architected by Cyberdyne Systems. No fate but what we make.</i>
</p>
