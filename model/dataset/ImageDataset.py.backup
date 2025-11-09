import os
from glob import glob
import random
import re
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode
from .util import *


class ImageDataset(Dataset):
    def __init__(self, root, in_image_size=256, out_image_size=256, shuffle=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64, load_keypoint=False, local_dir=None):
        super().__init__()
        self.image_loader = ["rgb.png", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.metadata_loader = ["metadata.json", metadata_loader]
        if local_dir is not None:
            root = copy_data_to_local(root, local_dir)
        self.all_img_suffixes = []
        self.samples = self._parse_folder(root)
        if shuffle:
            random.shuffle(self.samples)
        else:
            self.samples.sort()
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        self.load_keypoint = load_keypoint
        if load_keypoint:
            self.keypoint_loader = ["keypoint.txt", keypoint_loader]
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.load_background = load_background
        self.random_xflip = random_xflip

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '**/*'+image_path_suffix), recursive=True))
        if '*' in image_path_suffix:
            all_suffixes = set()
            base_pattern = image_path_suffix.split('*')[0]
            regex = re.escape(base_pattern) + r'.*'
            for path in result:
                match = re.search(regex, path)
                if match and not path.endswith(".mp4"):
                    all_suffixes.add(match.group(0))
            combined_pattern = '|'.join([re.escape(suffix) for suffix in all_suffixes])
            result = list(set([re.sub(combined_pattern, "{}", p) for p in result for suffix in all_suffixes]))
            self.all_img_suffixes = list(all_suffixes)
        else:
            result = [p.replace(image_path_suffix, '{}') for p in result]
            self.all_img_suffixes = [image_path_suffix]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.samples)
    
    def set_random_xflip(self, random_xflip):
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        for suffix in self.all_img_suffixes:
            try:
                self.image_loader[0] = suffix
                images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
                break
            except FileNotFoundError:
                continue
        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
        mask_dt = compute_distance_transform(masks)
        metadata = self._load_ids(path, self.metadata_loader)
        global_frame_id = torch.LongTensor([int(metadata.get("video_frame_id"))])
        xmin, ymin, xmax, ymax = metadata.get("crop_box_xyxy")
        full_w, full_h = metadata.get("video_frame_width"), metadata.get("video_frame_height")
        bboxs = torch.Tensor(
            [global_frame_id.item(), xmin, ymin, xmax - xmin, ymax - ymin, full_w, full_h, 0]
        ).unsqueeze(0)
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        flows = None
        if self.load_background:
            bg_fpath = os.path.join(os.path.dirname(path), 'background_frame.jpg')
            assert os.path.isfile(bg_fpath)
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.out_image_size, self.out_image_size))
        else:
            bg_images = None
        if self.load_dino_feature:
            dino_features = self._load_ids(path, self.dino_feature_loader, transform=torch.FloatTensor).unsqueeze(0)
        else:
            dino_features = None
        if self.load_dino_cluster:
            dino_clusters = self._load_ids(path, self.dino_cluster_loader, transform=transforms.ToTensor()).unsqueeze(0)
        else:
            dino_clusters = None
        if self.load_keypoint:
            keypoint = self._load_ids(path, self.keypoint_loader).unsqueeze(0)
            keypoint = keypoint / self.in_image_size * 2 - 1
        else:
            keypoint = None
        seq_idx = torch.LongTensor([index])
        frame_idx = torch.LongTensor([0])

        ## random horizontal flip
        if self.random_xflip and np.random.rand() < 0.5:
            xflip = lambda x: None if x is None else x.flip(-1)
            images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (*map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
            bboxs = horizontal_flip_box(bboxs)  # NxK

        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint, seq_idx, frame_idx)),)  # for batch collation
        return out
