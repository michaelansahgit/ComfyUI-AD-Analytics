import sys
import subprocess
import importlib.util
import os
import glob
import pickle
import numpy as np
import torch
import folder_paths
from PIL import Image
import io

# --- Dependency Check (Auto-Install) ---
def check_and_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        try:
            print(f"['AD_Analytics'] Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        except subprocess.CalledProcessError:
            pass

check_and_install("umap-learn", "umap")
check_and_install("matplotlib")

import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ==============================================================================
# NODE 1: AD_Data_Collector 
# Purpose: Saves the Image + Embedding pairs required for analysis
# ==============================================================================
class AD_Data_Collector:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "embedding": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "sample"}),
                # Default changed to keep project organized
                "dataset_path": ("STRING", {"default": "driving_dataset"}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "save_pair"
    CATEGORY = "Autonomous Driving/Analytics"
    OUTPUT_NODE = True

    def save_pair(self, images, embedding, filename_prefix, dataset_path):
        # 1. Setup paths
        base_output_dir = folder_paths.get_output_directory()
        target_dir = os.path.join(base_output_dir, dataset_path)
        os.makedirs(target_dir, exist_ok=True)

        # 2. Get counter and filename
        full_output_folder, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(
            filename_prefix, target_dir, images[0].shape[1], images[0].shape[0]
        )

        # 3. Save Files (Syncing Image and PKL)
        file_base = f"{filename}_{counter:05d}_"
        
        # Save PKL (The Latent Data)
        pkl_name = f"{file_base}.pkl"
        pkl_path = os.path.join(full_output_folder, pkl_name)
        with open(pkl_path, "wb") as f:
            # Handle ComfyUI Conditioning Wrapper
            data_to_save = embedding
            if isinstance(embedding, list) and len(embedding) > 0:
                 # Often conditioning is [[tensor, dict]], we want just the tensor if possible
                 # But sticking to raw dump is safer for maximum compatibility
                 pass 
            pickle.dump(data_to_save, f)

        # Save PNG (The Visual Reference)
        img_name = f"{file_base}.png"
        img_path = os.path.join(full_output_folder, img_name)
        
        img_tensor = images[0] 
        img_array = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img.save(img_path)

        print(f"['AD_Collector'] Saved pair: {pkl_name} & {img_name}")
        return (target_dir,)

# ==============================================================================
# NODE 2: AD_Latent_Visualizer (The UMAP Analysis Tool)
# Purpose: Loads the folders created by Node 1 and plots the distribution
# ==============================================================================
class AD_Latent_Visualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "driving_dataset"}),
                "n_neighbors": ("INT", {"default": 15, "min": 2, "max": 200}),
                "min_dist": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "random_state": ("INT", {"default": 42, "min": 0, "max": 10000}),
                "figure_size": ("INT", {"default": 12, "min": 5, "max": 50}),
                "thumbnail_zoom": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_umap"
    CATEGORY = "Autonomous Driving/Analytics"

    def generate_umap(self, path, n_neighbors, min_dist, random_state, figure_size, thumbnail_zoom):
        # Handle relative paths from ComfyUI output
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.get_output_directory(), path)

        all_embeddings = []
        loaded_images = [] 
        
        # 1. Gather Files
        pkl_files = glob.glob(os.path.join(path, "*.pkl"))
        if not pkl_files:
            return self.create_error_image(f"No .pkl files found in: {path}")

        print(f"['AD_Viz'] Found {len(pkl_files)} samples. Processing...")

        # 2. Load Data & Matching Images
        for pkl_path in pkl_files:
            try:
                # A. Load Embedding
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                # B. Normalize (Extract Tensor from ComfyUI Conditioning)
                emb = self.normalize_embedding(data)
                if emb is None: continue

                # C. Find Matching Image
                base_name = os.path.splitext(pkl_path)[0]
                image_path = None
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    if os.path.exists(base_name + ext):
                        image_path = base_name + ext
                        break
                
                # D. Load Image Thumbnail
                img_obj = None
                if image_path:
                    try:
                        with Image.open(image_path) as img:
                            img = img.convert("RGB")
                            img.thumbnail((256, 256)) 
                            img_obj = np.array(img)
                    except Exception:
                        pass

                # E. Add to Dataset
                for i in range(emb.shape[0]):
                    all_embeddings.append(emb[i])
                    loaded_images.append(img_obj)

            except Exception as e:
                print(f"Skipped {pkl_path}: {e}")

        if not all_embeddings:
            return self.create_error_image("No valid embeddings extracted.")

        # 3. UMAP Reduction
        full_data = np.array(all_embeddings)
        
        # Safety for small datasets
        actual_neighbors = min(n_neighbors, full_data.shape[0] - 1)
        if actual_neighbors < 2: actual_neighbors = 2
        
        init_mode = 'spectral'
        if full_data.shape[0] < 15: init_mode = 'random'

        reducer = umap.UMAP(
            n_components=2, 
            random_state=random_state, 
            n_neighbors=actual_neighbors, 
            min_dist=min_dist, 
            init=init_mode
        )
        
        try:
            coords = reducer.fit_transform(full_data)
        except Exception as e:
            return self.create_error_image(f"UMAP Error: {str(e)}")

        # 4. Plotting
        plt.figure(figsize=(figure_size, figure_size))
        ax = plt.gca()
        
        # Auto-scale axes
        if coords.shape[0] > 0:
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            margin = 0.1 * max(x_max - x_min, y_max - y_min) + 0.1
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
        
        # Draw Images
        for i, (x, y) in enumerate(coords):
            img_arr = loaded_images[i]
            if img_arr is not None:
                im = OffsetImage(img_arr, zoom=thumbnail_zoom)
                ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                ax.add_artist(ab)
            else:
                ax.scatter(x, y, c='red', alpha=0.5)

        plt.title(f'AD Dataset Distribution ({len(all_embeddings)} samples)')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()

        # 5. Save to Tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        pil_image = Image.open(buf).convert('RGB')
        return (torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0),)

    def normalize_embedding(self, data):
        # Logic to unwrap ComfyUI's complex Conditioning format
        if isinstance(data, list) and len(data) == 1: data = data[0]
        if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[1], dict):
            data = data[0] # Take tensor, ignore dict
        
        if hasattr(data, "detach"): # is tensor
            data = data.detach().cpu().numpy()
        
        if isinstance(data, np.ndarray):
            if data.ndim > 1: data = data.flatten()
            return data.reshape(1, -1)
        return None

    def create_error_image(self, message):
        print(f"['AD_Visualizer_Error'] {message}")
        # Create a black image with text would be better, but return black for now
        img = Image.new('RGB', (512, 512), color='black')
        return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)


