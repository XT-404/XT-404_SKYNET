import torch
import numpy as np

class Wan_Neural_Motion_Path:
    """
    MODULE E : NEURAL MOTION PATH (V1 OMEGA)
    Architecture : Latent Velocity Vector Generator
    Fonction : Injection de trajectoires forcées dans le flux de diffusion.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_x": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "motion_y": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "zoom": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "flow_persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MOTION_DATA",)
    FUNCTION = "generate_path"
    CATEGORY = "XT-404/V2_Omega"

    def generate_path(self, motion_x, motion_y, zoom, rotation, strength, flow_persistence):
        # On encapsule les paramètres de physique latente
        motion_data = {
            "vector": torch.tensor([motion_x, motion_y]),
            "zoom": zoom,
            "rotation": rotation,
            "strength": strength,
            "persistence": flow_persistence,
            "active": True if abs(motion_x) + abs(motion_y) + abs(zoom) + abs(rotation) > 0 else False
        }
        return (motion_data,)

NODE_CLASS_MAPPINGS = {"Wan_Neural_Motion_Path": Wan_Neural_Motion_Path}