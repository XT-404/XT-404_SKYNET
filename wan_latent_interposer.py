import torch

class Wan_Latent_Color_Lock:
    """
    MODULE F : LATENT COLOR LOCK (ANTI-DRIFT)
    Force la cohérence colorimétrique directement dans l'espace latent (avant décodage).
    Empêche la peau de devenir blanche/grise sur les longues vidéos.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "lock_latent_color"
    CATEGORY = "XT-404/V2_Omega"

    def get_statistics(self, x):
        # Calcule la moyenne et l'écart-type sur les dimensions spatiales (H, W)
        # x shape: [B, C, T, H, W] ou [B, C, H, W]
        if x.ndim == 5:
            # Pour la vidéo Wan: [Batch, Channel, Time, Height, Width]
            # On calcule les stats pour chaque Frame (Time) indépendamment
            # axes = (3, 4) -> H, W
            mean = x.mean(dim=(3, 4), keepdim=True)
            std = x.std(dim=(3, 4), keepdim=True)
        else:
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
        return mean, std

    def lock_latent_color(self, latent, strength):
        samples = latent["samples"].clone()
        
        # Si c'est une image unique ou vide, on ne fait rien
        if samples.ndim < 4: 
            return (latent,)

        # 1. Extraction de la Référence (Frame 0)
        # Wan Latent : [Batch, 16, Time, H, W]
        if samples.ndim == 5:
            ref_frame = samples[:, :, 0:1, :, :] # Frame 0
        else:
            ref_frame = samples # Pas de temps, fallback
            
        # 2. Calcul des Stats de la Référence (L'ADN couleur de l'image de base)
        mu_ref, sigma_ref = self.get_statistics(ref_frame)
        
        # 3. Calcul des Stats de la Cible (Toute la vidéo)
        mu_tgt, sigma_tgt = self.get_statistics(samples)
        
        # 4. AdaIN (Adaptive Instance Normalization) dans l'espace Latent
        # On aligne la distribution de chaque frame sur la frame 0
        # Formule : (x - mean_curr) / std_curr * std_ref + mean_ref
        target_normalized = (samples - mu_tgt) / (sigma_tgt + 1e-6)
        samples_corrected = target_normalized * sigma_ref + mu_ref
        
        # 5. Application (Lerp)
        # On mélange l'original et la correction selon la force demandée
        final_samples = torch.lerp(samples, samples_corrected, strength)
        
        # On remet la Frame 0 intacte par sécurité (bien que mathématiquement identique)
        if samples.ndim == 5:
            final_samples[:, :, 0, :, :] = samples[:, :, 0, :, :]
            
        return ({"samples": final_samples},)

NODE_CLASS_MAPPINGS = {"Wan_Latent_Color_Lock": Wan_Latent_Color_Lock}
NODE_DISPLAY_NAME_MAPPINGS = {"Wan_Latent_Color_Lock": "Wan Latent Color Lock (Anti-Drift)"}