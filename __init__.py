from .nodes import AD_Data_Collector, AD_Latent_Visualizer

NODE_CLASS_MAPPINGS = {
    "AD_Data_Collector": AD_Data_Collector,
    "AD_Latent_Visualizer": AD_Latent_Visualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AD_Data_Collector": "AD Data Collector (Save)",
    "AD_Latent_Visualizer": "AD Latent Explorer (View)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']