#!/usr/bin/env python3
"""
Extract DINO features (feat16.png) for mouse dataset

Based on 3DAnimals codebase decoding logic (model/dataset/util.py)
"""

import os
import sys
import numpy as np
from PIL import Image
from einops import rearrange
import torch
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def encode_feat_to_img(feat, n_channels=16):
    """
    Encode feature map to image (reverse of dencode_feat_from_img from util.py)

    Args:
        feat: numpy array of shape (C, H, W) with C=16 channels
        n_channels: number of feature channels (16)

    Returns:
        PIL Image of shape (H, tiles*W, 3)
    """
    # Transpose to HxWxC
    feat = feat.transpose(1, 2, 0)  # CxHxW -> HxWxC

    # Quantize to uint8
    feat_uint8 = (feat * 255).astype(np.uint8)

    # Calculate padding
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    n_tiles = int((n_channels + n_addon_channels) / 3)

    # Add padding channels
    if n_addon_channels > 0:
        padding = np.zeros((feat.shape[0], feat.shape[1], n_addon_channels), dtype=np.uint8)
        feat_padded = np.concatenate([feat_uint8, padding], axis=2)
    else:
        feat_padded = feat_uint8

    # Rearrange to tiled image
    img = rearrange(feat_padded, 'h w (t c) -> h (t w) c', t=n_tiles, c=3)

    return Image.fromarray(img)


class DINOFeatureExtractor:
    def __init__(self, model_name='dino_vits8', feature_dim=16, device='cuda'):
        """
        Initialize DINO feature extractor

        Args:
            model_name: DINO model variant (dino_vits8, dino_vits16, dino_vitb8, dino_vitb16)
            feature_dim: Output feature dimension after PCA (default: 16)
            device: 'cuda' or 'cpu'
        """
        self.feature_dim = feature_dim
        self.device = device

        print(f"Loading DINO model: {model_name}...")
        self.dino_model = torch.hub.load('facebookresearch/dino:main', model_name)
        self.dino_model.eval()
        self.dino_model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # PCA will be fitted on first batch
        self.pca = None
        print("✅ DINO model loaded")

    def extract_raw_features(self, rgb_image_path):
        """
        Extract raw DINO features from RGB image

        Returns:
            features: numpy array of shape (H, W, D) where D is raw DINO dim
        """
        # Load and preprocess image
        image = Image.open(rgb_image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get image size after transform
        _, _, img_h, img_w = img_tensor.shape

        # Calculate expected patch grid size
        # ViT-S/8 uses patch_size=8, so grid = img_size // patch_size
        patch_size = 8  # for dino_vits8
        h = img_h // patch_size
        w = img_w // patch_size

        # Extract DINO features
        with torch.no_grad():
            features = self.dino_model.get_intermediate_layers(img_tensor, n=1)[0]
            # features shape: [1, num_patches+1, feature_dim]
            patch_features = features[:, 1:, :]  # Remove CLS token

            # Reshape to spatial grid
            num_patches = patch_features.shape[1]
            expected_patches = h * w

            if num_patches != expected_patches:
                # Handle non-square or different size images
                # Try to infer the correct grid size
                h = w = int(np.sqrt(num_patches))
                if h * w != num_patches:
                    raise ValueError(f"Cannot reshape {num_patches} patches to square grid. "
                                     f"Image size: {img_h}x{img_w}, expected patches: {expected_patches}")

            spatial_features = patch_features.reshape(1, h, w, -1)

        return spatial_features.squeeze(0).cpu().numpy()  # (H, W, D)

    def fit_pca(self, feature_samples, n_samples=100):
        """
        Fit PCA on a batch of feature samples

        Args:
            feature_samples: list of raw feature arrays
            n_samples: number of samples to use for PCA fitting
        """
        from sklearn.decomposition import PCA

        print(f"Fitting PCA on {len(feature_samples)} samples...")

        # Concatenate all features
        all_features = []
        for feat in feature_samples[:n_samples]:
            all_features.append(feat.reshape(-1, feat.shape[-1]))

        all_features = np.concatenate(all_features, axis=0)

        # Fit PCA
        self.pca = PCA(n_components=self.feature_dim)
        self.pca.fit(all_features)

        variance_ratio = self.pca.explained_variance_ratio_.sum()
        print(f"✅ PCA fitted. Explained variance: {variance_ratio:.3f}")

    def reduce_features(self, raw_features):
        """
        Apply PCA to reduce feature dimensions

        Args:
            raw_features: (H, W, D) numpy array

        Returns:
            reduced_features: (H, W, feature_dim) numpy array
        """
        h, w, d = raw_features.shape
        flat_features = raw_features.reshape(-1, d)

        reduced_flat = self.pca.transform(flat_features)
        reduced_spatial = reduced_flat.reshape(h, w, self.feature_dim)

        return reduced_spatial

    def extract_and_save(self, rgb_image_path, output_path):
        """
        Complete pipeline: RGB → DINO features → feat16.png

        Args:
            rgb_image_path: Path to RGB image
            output_path: Path to save feat16.png
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted! Call fit_pca() first.")

        # 1. Extract raw DINO features
        raw_features = self.extract_raw_features(rgb_image_path)  # (H, W, D)

        # 2. PCA reduction
        reduced_features = self.reduce_features(raw_features)  # (H, W, 16)

        # 3. Upsample to 256×256
        upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(reduced_features).permute(2, 0, 1).unsqueeze(0).float(),
            size=(256, 256), mode='bilinear', align_corners=False
        )
        feat_map = upsampled.squeeze(0).cpu().numpy()  # (16, 256, 256)

        # 4. Normalize to [0, 1] range (per-channel normalization)
        for c in range(feat_map.shape[0]):
            feat_min = feat_map[c].min()
            feat_max = feat_map[c].max()
            if feat_max > feat_min:
                feat_map[c] = (feat_map[c] - feat_min) / (feat_max - feat_min)
            else:
                feat_map[c] = 0.5  # Constant channel

        # 5. Encode and save
        img = encode_feat_to_img(feat_map, n_channels=self.feature_dim)
        img.save(output_path)


def process_mouse_dataset(data_dir, output_dir=None, model_name='dino_vits8', device='cuda'):
    """
    Process entire mouse dataset to extract DINO features

    Args:
        data_dir: Path to mouse dataset (e.g., fauna_mouse/large_scale/mouse_dannce_6view)
        output_dir: Output directory (default: same as data_dir, in-place)
        model_name: DINO model variant
        device: 'cuda' or 'cpu'
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)

    # Find all RGB images
    rgb_files = sorted(data_dir.glob('**/*_rgb.png'))

    if len(rgb_files) == 0:
        print(f"❌ No RGB images found in {data_dir}")
        return

    print(f"Found {len(rgb_files)} RGB images")

    # Initialize extractor
    extractor = DINOFeatureExtractor(model_name=model_name, feature_dim=16, device=device)

    # First pass: collect samples for PCA fitting
    print("\n[1/2] Collecting samples for PCA fitting...")
    n_pca_samples = min(100, len(rgb_files))
    raw_feature_samples = []

    for rgb_file in tqdm(rgb_files[:n_pca_samples], desc="Extracting samples"):
        raw_features = extractor.extract_raw_features(str(rgb_file))
        raw_feature_samples.append(raw_features)

    # Fit PCA
    extractor.fit_pca(raw_feature_samples, n_samples=n_pca_samples)

    # Second pass: extract and save all features
    print(f"\n[2/2] Extracting DINO features for all {len(rgb_files)} images...")

    for rgb_file in tqdm(rgb_files, desc="Extracting features"):
        # Determine output path
        feat_file = str(rgb_file).replace('_rgb.png', '_feat16.png')

        # Skip if already exists
        if os.path.exists(feat_file):
            continue

        # Extract and save
        try:
            extractor.extract_and_save(str(rgb_file), feat_file)
        except Exception as e:
            print(f"\n⚠️ Failed to process {rgb_file}: {e}")
            continue

    print(f"\n✅ DINO feature extraction complete!")
    print(f"Processed {len(rgb_files)} images")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract DINO features for mouse dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to mouse dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as data_dir)')
    parser.add_argument('--model', type=str, default='dino_vits8',
                       choices=['dino_vits8', 'dino_vits16', 'dino_vitb8', 'dino_vitb16'],
                       help='DINO model variant (default: dino_vits8)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    process_mouse_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device
    )


if __name__ == '__main__':
    main()
