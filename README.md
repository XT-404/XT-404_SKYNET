# 🤖 XT-404 Skynet Suite: Wan 2.2 Integration
### *The "Omega Edition" for ComfyUI*

![Skynet Banner](https://img.shields.io/badge/SYSTEM-ONLINE-green?style=for-the-badge) 
![Version](https://img.shields.io/badge/VERSION-3.8%20GOLD-blue?style=for-the-badge) 
![Architecture](https://img.shields.io/badge/ARCH-XT--404-red?style=for-the-badge)
![License](https://img.shields.io/badge/LICENSE-MIT-black?style=for-the-badge)

La **XT-404 Skynet Suite** est une collection de nœuds personnalisés pour ComfyUI, conçue avec une précision chirurgicale pour les modèles de diffusion vidéo **Wan 2.1 et 2.2**. 

Contrairement aux solutions génériques, cette suite privilégie la "Suprématie Visuelle" : atteindre une qualité 8K de grade OLED grâce à des algorithmes heuristiques qui protègent l'intégrité du signal et optimisent la VRAM.

---

## ⚡ Caractéristiques Clés

*   **Zero-Point Noise Injection :** Élimine le "bruit de neige" statique dès le premier échantillonnage.
*   **ARRI Rolloff Tone Mapping :** Gestion cinématographique des hautes lumières pour éviter le clipping.
*   **Nano-Repair (Genisys) :** Surveillance des tenseurs en temps réel pour prévenir les écrans noirs (NaNs).
*   **Sentinel Telemetry :** Monitoring console ("Le Mouchard") analysant la saturation et la VRAM à chaque étape.

---

## 🛠️ Prérequis & Dépendances

### Système
* **ComfyUI :** Dernière version recommandée.
* **Python :** 3.10 ou supérieur.
* **FFmpeg :** Requis pour le nœud Compressor (via `imageio-ffmpeg`).

### Modèles Wan 2.2
Assurez-vous de posséder les fichiers suivants :
* VAE, CLIP, et UNet/Transformer (Safetensors ou GGUF).

> [!CAUTION]
> **PROTOCOLE D'INFILTRATION (GGUF) :**
> Pour utiliser les modèles quantifiés GGUF via le **Cyberdyne Model Hub**, l'extension **ComfyUI-GGUF** est **OBLIGATOIRE**.
> 📥 **Source :** `city96/ComfyUI-GGUF`
> *Sans cela, le Hub fonctionnera uniquement en mode Safetensors.*

---

## 📦 Installation

1. Accédez à votre dossier `custom_nodes` :
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clonez le dépôt :
   ```bash
   git clone https://github.com/YourUsername/XT-404-Skynet-Suite.git
   ```
3. Installez les dépendances :
   ```bash
   pip install imageio-ffmpeg scikit-image
   ```

---

## 🏗️ Architecture de la Suite

### 1. The Core Engine (`XT404_Skynet_Nodes.py`)
Remplace le KSampler standard par un moteur hybride optimisé pour le *Flow Matching* de Wan.
* **Nodes :** Master, Chain, Refiner.
* **Innovation :** Calculateur de Sigma spécifique aux formules de Wan 2.1/2.2.

### 2. Universal Loader (`cyberdyne_model_hub.py`)
Un chargeur unifié intelligent pour Checkpoints, SafeTensors et GGUF.
* **Smart Offload :** Décharge agressive des modèles inutilisés vers la RAM système pour libérer la VRAM.
* **Checksum Verification :** Vérification SHA256 pour garantir l'intégrité des modèles lourds (30GB+).

### 3. Visual Supremacy Suite (`wan_visual_supremacy.py`)
Le pipeline post-traitement pour éliminer le "look plastique" de l'IA.
* **Temporal Lock Pro :** Stabilisateur post-décodage qui réduit le scintillement (flicker).
* **OLED Dynamix :** Mappage de ton logarithmique pour des noirs profonds et des textures organiques.

### 4. Nano-Repair System (`wan_genisys.py`)
* **Node :** `Cyberdyne Genisys [OMNISCIENT]`
* **Fonction :** Enveloppe l'UNET pour détecter la dérive des tenseurs. Si une valeur tend vers l'infini, elle est clampée immédiatement pour éviter le crash du rendu.

### 5. T-X Interpolator (`wan_tx_node.py`)
Génère une transition entre une image de début et de fin.
* **Inverse Structural Repulsion :** Injecte du bruit haute fréquence dérivé des différences latentes pour forcer le modèle à "halluciner" une transformation fluide.

---

## 🎛️ Stratégie de Workflow Recommandée

Pour obtenir le rendu "8K OLED" ultime, connectez les nœuds dans cet ordre :

1.  **Input :** `Cyberdyne Model Hub` → `Wan Text/Vision Cache`.
2.  **Génération :** `WanImageToVideoUltra` → `XT-404 Skynet 1 (Master)`.
3.  **Décodage :** `VAE Decode`.
4.  **Traitement Skynet :**
    *   `Temporal Lock Pro` (Stabilisation).
    *   `OLED Dynamix` (Sculpture de la lumière).
    *   `Organic Skin` (Grain pellicule).
5.  **Finalisation :** `Wan Chroma Mimic` (Validation du signal) → `Wan Compressor`.

---

## 📟 Console HUD (XT-Mouchard)

Surveillez votre console durant le rendu :
* 🟢 **VERT :** Signal sain.
* 🟡 **JAUNE :** Signal fort détecté (Rolloff actif).
* 🔴 **ROUGE :** Saturation critique / Clipping (Réduisez le `specular_pop`).

**Exemple de log :**
```text
[XT-MIMIC] 🎨 FINAL VALIDATION | DynRange: [0.000, 0.982]
   └── Signal Integrity: OK (Clip: 0.00%)
```

---

## 📜 Crédits & Vision

*   **Architecte :** XT-404 Omega
*   **Organisation :** Cyberdyne Systems
*   **Statut :** GOLD MASTER (V3.8)

> *"There is no fate but what we make."*

---
*Dépôt maintenu par l'unité de recherche Cyberdyne. Pour tout bug, ouvrez une "Infiltration Report" (Issue).*
