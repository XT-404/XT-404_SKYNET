# 🤖 XT-404 Skynet Suite: Wan 2.2 Integration
### *The "Omega Edition" for ComfyUI*

![Skynet Banner](https://img.shields.io/badge/SYSTEM-ONLINE-green?style=for-the-badge) 
![Version](https://img.shields.io/badge/VERSION-3.9%20OMEGA-blue?style=for-the-badge) 
![Architecture](https://img.shields.io/badge/ARCH-XT--404-red?style=for-the-badge)
![License](https://img.shields.io/badge/LICENSE-MIT-black?style=for-the-badge)

The **XT-404 Skynet Suite** is a highly specialized, battle-tested collection of custom nodes for ComfyUI, specifically engineered for **Wan 2.1 and 2.2** video diffusion models.

Unlike generic implementations, this suite focuses on **"Visual Supremacy"**—achieving 8K, OLED-grade quality with mathematical precision. It abandons standard processing for heuristic, context-aware algorithms that protect signal integrity, manage VRAM surgically, and eliminate digital artifacts.

---

## ⚡ Key Features (Omega Edition)

*   **🛡️ Temporal Shield (T-1000):** A polymetric structural anchor that prevents the model from hallucinating or drifting too far from the reference during long generations.
*   **⚛️ Neural Motion Physics:** Injects forced latent velocity vectors to simulate camera movements (Pan, Tilt, Zoom, Rotate) without complex prompting.
*   **🌈 Spectre Chroma Filter:** Uses **FFT (Fast Fourier Transform)** to lock color frequencies, eliminating the "Rainbow/Irisation" flickering common in Wan models.
*   **📈 Infiltration Upscaler:** A true Spatio-Temporal VAE upscaler that processes video in chunks with temporal feathering for seamless high-res reconstruction.
*   **💀 Genisys Nano-Repair:** Real-time tensor monitoring that clamps values to prevent "Black Screen" (NaNs) in TF32/BF16 modes.
*   **🧹 Memory Salvation:** Surgical VRAM/RAM purging using OS-level APIs to prevent crashes on consumer GPUs.

---

## ⚠️ Requirements

*   **ComfyUI:** Latest version.
*   **Wan 2.2 Models:** VAE, CLIP, and UNet/Transformer (Safetensors or GGUF).
*   **Python:** 3.10+.
*   **FFmpeg:** Required for the Compressor node (installed via `imageio-ffmpeg`).

> [!CAUTION]
> **INFILTRATION PROTOCOL (GGUF):**
> To utilize **GGUF Quantized Models** with the `Cyberdyne Model Hub`, the **ComfyUI-GGUF** engine is **REQUIRED**.
> 
> 📥 **Download Engine:** `city96/ComfyUI-GGUF`
>
> *Without this engine, the Model Hub will operate in Safetensors-only mode.*

---

## 📦 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```

2.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/XT-404-Skynet-Suite.git
    ```

3.  Install dependencies:
    ```bash
    pip install imageio-ffmpeg scikit-image
    ```

---

## 🛠️ Module Breakdown

### 1. The Core Engine & Loaders
*   **XT-404 Skynet (Master/Chain/Refiner):** Replaces standard KSamplers. Features a **Zero-Point Noise Injection** system (0 + Noise = Pure Noise) to eliminate static "snow" artifacts.
*   **Cyberdyne Model Hub:** A unified loader for Checkpoints and GGUF. Features **Smart Offload** to aggressively free VRAM for the sampler.
*   **Wan Text/Vision Cache:** "OneShot" caching systems that pin embeddings to RAM, preventing redundant encoding steps.

### 2. Generators & Interpolators
*   **Wan ImageToVideo Ultra (Hybrid V19):** The flagship generator. Includes a "Smart VRAM Scanner" that automatically adjusts tile sizes (512/1024/1280) based on available GPU memory.
*   **Wan T-X Interpolator:** Uses **Inverse Structural Repulsion** to force the model to hallucinate a transformation path between a Start and End image, rather than a simple cross-fade.

### 3. The "Visual Supremacy" Stack
*   **OLED Dynamix (ARRI Rolloff):** Applies a logarithmic tone curve with shoulder compression to preserve highlight details (prevents white clipping) while crushing blacks for OLED displays.
*   **Temporal Lock Pro:** A post-decode stabilizer that blends low-delta pixels to eliminate background flicker.
*   **Organic Skin:** Adds resolution-aware film grain that respects skin tones (luma masking).

### 4. Experimental Protocols (Omega)
*   **Wan Neural Motion Path:** Defines physics vectors (`motion_x`, `motion_y`, `zoom`, `rotation`) to drive the generation.
*   **Wan Spectre Chroma:** Analyzes the U/V channels of the video via FFT and dampens frequency drift to stop color flashing.
*   **Wan Infiltration Upscaler:** Upscales latent chunks with temporal overlaps to create 4K video without VRAM overflows.

### 5. System Utilities
*   **Wan Cycle Terminator:** The "Nuclear Option" for memory. Uses Windows API `EmptyWorkingSet` to force-release RAM and VRAM.
*   **Wan Video Compressor:** A thread-safe H.265 encoder optimized for high-core count CPUs (prevents UI freezing).

---

## 🎛️ Recommended Workflow Strategy

For the ultimate **8K OLED** result, chain the nodes in this specific order:

1.  **Loader:** `Cyberdyne Model Hub` (Load Model & VAE).
2.  **Prompt:** `Wan Text Cache` & `Wan Vision Cache`.
3.  **Physics (Optional):** `Wan Neural Motion Path` (Inject movement vectors).
4.  **Generation:** `WanImageToVideoUltra` → `XT-404 Skynet 1 (Master)`.
    *   *Enable `temporal_shield` (0.2 - 0.5) for long videos.*
5.  **Refinement:** `XT-404 Skynet 3 (Refiner)` (Denoise ~0.3).
6.  **Upscale (Optional):** `Wan Infiltration Upscaler`.
7.  **Decode:** `VAE Decode`.
8.  **Post-Processing:**
    *   `Wan Spectre Chroma` (Stabilize Colors).
    *   `OLED Dynamix` (Tone Map).
    *   `Organic Skin` (Texture).
9.  **Encode:** `Wan Video Compressor`.
10. **Cleanup:** `Wan Cycle Terminator`.

---

## 📟 The Console HUD (XT-Mouchard)

Do not ignore the console! The suite communicates signal health in real-time via the **Omniscient HUD**.

*   🟢 **GREEN:** Signal is healthy.
*   🟡 **YELLOW:** High signal detected (Rolloff is active).
*   🔴 **RED:** Critical saturation/clipping (Lower `specular_pop` or check VAE).

**Example Genisys Log:**
```text
ST:04 │ INJECT │ WARMUP SEQ   │ T:0.012 M:0.045 │ Δ:0.021/0.025 [|||.......] │ SIG:▰▰▰▰▰▱▱▱▱▱ │ 🔒 LOCKED
```
---

# 🤖 XT-404 Skynet Suite: Intégration Wan 2.2
### *L'édition "Omega" pour ComfyUI*

![Skynet Banner](https://img.shields.io/badge/SYSTEM-ONLINE-green?style=for-the-badge) 
![Version](https://img.shields.io/badge/VERSION-3.9%20OMEGA-blue?style=for-the-badge) 
![Architecture](https://img.shields.io/badge/ARCH-XT--404-red?style=for-the-badge)
![License](https://img.shields.io/badge/LICENSE-MIT-black?style=for-the-badge)

La **XT-404 Skynet Suite** est une collection de nœuds de combat hautement spécialisés pour ComfyUI, conçue avec une précision chirurgicale pour les modèles de diffusion vidéo **Wan 2.1 et 2.2**.

Contrairement aux implémentations génériques, cette suite vise la **"Suprématie Visuelle"** : atteindre une qualité 8K de grade OLED avec une précision mathématique. Elle abandonne le traitement standard pour des algorithmes heuristiques conscients du contexte, qui protègent l'intégrité du signal, gèrent la VRAM de manière chirurgicale et éliminent les artefacts numériques.

> *"There is no fate but what we make."*

---

## ⚡ Caractéristiques Clés (Édition Omega)

*   **🛡️ Temporal Shield (T-1000) :** Une ancre structurelle polymétrique qui empêche le modèle "d'halluciner" ou de dériver trop loin de la référence initiale lors des longues générations.
*   **⚛️ Neural Motion Physics :** Injecte des vecteurs de vélocité latente forcés pour simuler des mouvements de caméra (Pan, Tilt, Zoom, Rotation) sans prompt complexe.
*   **🌈 Filtre Spectre Chroma :** Utilise la **FFT (Fast Fourier Transform)** pour verrouiller les fréquences colorimétriques, éliminant le scintillement "Arc-en-ciel/Irisation" fréquent sur les modèles Wan.
*   **📈 Infiltration Upscaler :** Un véritable upscaler VAE Spatio-Temporel qui traite la vidéo par paquets avec un fondu temporel (feathering) pour une reconstruction haute résolution sans coupures.
*   **💀 Nano-Réparation Genisys :** Monitoring des tenseurs en temps réel qui clampe les valeurs pour prévenir les "Écrans Noirs" (NaNs) en mode TF32/BF16.
*   **🧹 Memory Salvation :** Purge chirurgicale de la VRAM/RAM utilisant les APIs système (OS-level) pour prévenir les crashs sur les GPU grand public.

---

## ⚠️ Prérequis

*   **ComfyUI :** Dernière version recommandée.
*   **Modèles Wan 2.2 :** VAE, CLIP, et UNet/Transformer (Safetensors ou GGUF).
*   **Python :** 3.10+.
*   **FFmpeg :** Requis pour le nœud Compressor (installé via `imageio-ffmpeg`).

> [!CAUTION]
> **PROTOCOLE D'INFILTRATION (GGUF) :**
> Pour utiliser les **Modèles Quantifiés GGUF** avec le `Cyberdyne Model Hub`, le moteur **ComfyUI-GGUF** est **OBLIGATOIRE**.
> 
> 📥 **Télécharger le Moteur :** `city96/ComfyUI-GGUF`
>
> *Sans ce moteur, le Model Hub fonctionnera uniquement en mode Safetensors.*

---

## 📦 Installation

1.  Naviguez vers votre dossier `custom_nodes` ComfyUI :
    ```bash
    cd ComfyUI/custom_nodes/
    ```

2.  Clonez ce dépôt :
    ```bash
    git clone https://github.com/VotreNomUtilisateur/XT-404-Skynet-Suite.git
    ```

3.  Installez les dépendances :
    ```bash
    pip install imageio-ffmpeg scikit-image
    ```

---

## 🛠️ Analyse des Modules

### 1. Moteur Central & Chargeurs
*   **XT-404 Skynet (Master/Chain/Refiner) :** Remplace les KSamplers standards. Intègre un système **Zero-Point Noise Injection** (0 + Bruit = Bruit Pur) pour éliminer la "neige" statique.
*   **Cyberdyne Model Hub :** Un chargeur unifié pour Checkpoints et GGUF. Dispose du **Smart Offload** pour libérer agressivement la VRAM pour le sampler.
*   **Wan Text/Vision Cache :** Systèmes de cache "OneShot" qui épinglent les embeddings en RAM, évitant les ré-encodages redondants.

### 2. Générateurs & Interpolateurs
*   **Wan ImageToVideo Ultra (Hybrid V19) :** Le générateur amiral. Inclut un "Smart VRAM Scanner" qui ajuste automatiquement la taille des tuiles (512/1024/1280) selon la mémoire GPU disponible.
*   **Wan T-X Interpolator :** Utilise la **Répulsion Structurelle Inverse** pour forcer le modèle à halluciner un chemin de transformation entre une image de début et de fin, plutôt qu'un simple fondu enchaîné.

### 3. La Suite "Suprématie Visuelle"
*   **OLED Dynamix (ARRI Rolloff) :** Applique une courbe tonale logarithmique avec compression d'épaule pour préserver les détails des hautes lumières (anti-clipping) tout en écrasant les noirs pour les écrans OLED.
*   **Temporal Lock Pro :** Stabilisateur post-décodage qui fusionne les pixels à faible delta pour éliminer le scintillement d'arrière-plan.
*   **Organic Skin :** Ajoute du grain de film conscient de la résolution qui respecte les tons chair (masquage luma).

### 4. Protocoles Expérimentaux (Omega)
*   **Wan Neural Motion Path :** Définit des vecteurs physiques (`motion_x`, `motion_y`, `zoom`, `rotation`) pour piloter la génération.
*   **Wan Spectre Chroma :** Analyse les canaux U/V de la vidéo via FFT et amortit la dérive fréquentielle pour stopper les flashs de couleur.
*   **Wan Infiltration Upscaler :** Upscale des paquets latents avec chevauchement temporel pour créer de la vidéo 4K sans débordement VRAM.

### 5. Utilitaires Système
*   **Wan Cycle Terminator :** L'option "Nucléaire" pour la mémoire. Utilise l'API Windows `EmptyWorkingSet` pour forcer la libération de la RAM et VRAM.
*   **Wan Video Compressor :** Un encodeur H.265 thread-safe optimisé pour les CPU à grand nombre de cœurs (empêche le gel de l'interface).

---

## 🎛️ Stratégie de Workflow Recommandée

Pour obtenir le résultat **8K OLED** ultime, connectez les nœuds dans cet ordre précis :

1.  **Loader :** `Cyberdyne Model Hub` (Charger Modèle & VAE).
2.  **Prompt :** `Wan Text Cache` & `Wan Vision Cache`.
3.  **Physique (Optionnel) :** `Wan Neural Motion Path` (Injecter vecteurs de mouvement).
4.  **Génération :** `WanImageToVideoUltra` → `XT-404 Skynet 1 (Master)`.
    *   *Activez `temporal_shield` (0.2 - 0.5) pour les vidéos longues.*
5.  **Raffinement :** `XT-404 Skynet 3 (Refiner)` (Denoise ~0.3).
6.  **Upscale (Optionnel) :** `Wan Infiltration Upscaler`.
7.  **Décodage :** `VAE Decode`.
8.  **Post-Traitement :**
    *   `Wan Spectre Chroma` (Stabilisation Couleurs).
    *   `OLED Dynamix` (Tone Map).
    *   `Organic Skin` (Texture).
9.  **Encodage :** `Wan Video Compressor`.
10. **Nettoyage :** `Wan Cycle Terminator`.

---

## 📟 Console HUD (XT-Mouchard)

N'ignorez pas la console ! La suite communique la santé du signal en temps réel via le **HUD Omniscient**.

*   🟢 **VERT :** Signal sain.
*   🟡 **JAUNE :** Signal fort détecté (Rolloff actif).
*   🔴 **ROUGE :** Saturation critique / Clipping (Réduisez le `specular_pop` ou vérifiez le VAE).

**Exemple de Log Genisys :**
```text
ST:04 │ INJECT │ WARMUP SEQ   │ T:0.012 M:0.045 │ Δ:0.021/0.025 [|||.......] │ SIG:▰▰▰▰▰▱▱▱▱▱ │ 🔒 LOCKED
