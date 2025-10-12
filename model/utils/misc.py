import os
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import torchvision.utils as tvutils
import zipfile
from random import shuffle
from tqdm import tqdm
from subprocess import run
from einops import rearrange, repeat
import torch.nn.functional as F
from omegaconf.errors import ConfigAttributeError
from dataclasses import fields, is_dataclass
from ..render.obj import write_obj


def setup_runtime(cfg):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = cfg.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    # Setup random seeds for reproducibility
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Environment: GPU {cuda_device_id} - seed {seed}")
    return device


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth')),
            key=lambda x: int(''.join([c for c in x if c.isdigit()]))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                try:
                    os.remove(name)
                except FileNotFoundError:
                    pass


def archive_code(arc_path, filetypes=['.py']):
    print(f"Archiving code to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '[!results]*', '**', '*'+ftype), recursive=True))  # ignore results folder
        flist.extend(glob.glob(os.path.join(cur_dir, '*'+ftype)))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_videos(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.mp4', cycle=False):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0, 1, 3, 4, 2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        vid = cv2.VideoWriter(fpath, fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[..., ::-1] * 255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.png'):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0, 2, 3, 1)
    for i, img in enumerate(imgs):
        img = np.concatenate([np.flip(img[..., :3], -1), img[..., 3:]], -1)  # RGBA to BGRA
        if 'depth' in suffix:
            im_out = np.uint16(img * 65535.)
        else:
            im_out = np.uint8(img * 255.)

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        cv2.imwrite(fpath, im_out)


def save_txt(out_fold, data, prefix='', suffix='', fnames=None, ext='.txt', fmt='%.6f', delim=', '):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    for i, d in enumerate(data):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        np.savetxt(fpath, d, fmt=fmt, delimiter=delim)


def save_obj(out_fold, meshes=None, save_material=True, feat=None, prefix='', suffix='', fnames=None, resolution=[256, 256]):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    if meshes.v_pos is None:
        return

    batch_size = meshes.v_pos.shape[0]
    for i in range(batch_size):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ".obj"))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        write_obj(out_fold_i, prefix+fname+suffix, meshes, i, save_material=save_material, feat=feat, resolution=resolution)


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b, 1, 1)) ** 2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b, 1, 1)) ** 2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1 * n2).sum(3).clamp(-1, 1).acos() / np.pi * 180
    return dist * mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' % out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)


def image_grid(tensor, nrow=None):
    b, c, h, w = tensor.shape
    if nrow is None:
        nrow = int(np.ceil(b ** 0.5))
    if c == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tvutils.make_grid(tensor, nrow=nrow, normalize=False)
    return tensor


def video_grid(tensor, nrow=None):
    return torch.stack([image_grid(t, nrow=nrow) for t in tensor.unbind(1)], 0)


def in_range(x, range, default_indicator=None):
    range_min, range_max = range
    if isinstance(range_max, str):
        range_max = float(range_max)
    if isinstance(range_min, str):
        range_min = float(range_min)
    min_check = (x >= range_min)
    max_check = (x < range_max)
    if default_indicator is not None:
        if range_min == default_indicator:
            min_check = True
        if range_max == default_indicator:
            max_check = True
    return min_check and max_check


def load_cfg(self, cfg, config_class):
    """Load configs defined in config_class only and set attributes in self, recurse if a field is a dataclass"""
    cfg_dict = {}
    for field in fields(config_class):
        if is_dataclass(field.type):  # Recurse if field is dataclass
            value = load_cfg(None, getattr(cfg, field.name), field.type)
        else:
            try:
                value = getattr(cfg, field.name)
            except ConfigAttributeError:
                print(f"{config_class.__name__}.{field.name} not in config, using default value: {field.default}")
                continue
        cfg_dict[field.name] = value
    cfg = config_class(**cfg_dict)
    if self is not None:
        self.cfg = cfg
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
    return cfg


def add_text_to_image(img, text, pos=(12, 12), color=(1, 1, 1), font_scale=0.5, thickness=1):
    if isinstance(img, torch.Tensor):
        img = img.permute(1,2,0).cpu().numpy()
    # if grayscale -> convert to RGB
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
    img = cv2.putText(np.ascontiguousarray(img), text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def normalize_depth(depth, mask=None):
    # Normalize depth values to [0, 1] range based on the minimum and maximum depth values in masked regions
    # Set all background to 0
    if mask is not None:
        depth_for_min = torch.where(mask.bool(), depth, torch.full_like(depth, float('inf')))
        depth_for_max = torch.where(mask.bool(), depth, torch.full_like(depth, -float('inf')))
    else:
        depth_for_min = depth
        depth_for_max = depth
    depth_min = depth_for_min.amin(dim=(-1, -2), keepdim=True)
    depth_max = depth_for_max.amax(dim=(-1, -2), keepdim=True)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    if mask is not None:
        normalized_depth = torch.where(mask.bool(), normalized_depth, torch.zeros_like(normalized_depth))
    return normalized_depth


def images_to_video(image_files, output_path, fps=10):
    frame = cv2.imread(str(image_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    print(f"Saving video to {output_path}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        out.write(frame)
    out.release()


def validate_tensor_to_device(x, device=None):
    if isinstance(x, (list, tuple)):
        return x
    if torch.any(torch.isnan(x)):
        return None
    elif device is None:
        return x
    else:
        return x.to(device)


def validate_all_to_device(batch, device=None):
    return tuple(validate_tensor_to_device(x, device) for x in batch)


def indefinite_generator(loader):
    while True:
        for x in loader:
            yield x


def get_all_sequence_dirs(data_dir, image_suffix="rgb.png", sorted=False):
    result = set()
    for root, dirs, files in tqdm(os.walk(data_dir, followlinks=True), desc="Getting all sequence dirs"):
        for f in files:
            if f.endswith(image_suffix):
                result.add(root)
    result = list(result)
    if sorted:
        result.sort()
    else:
        shuffle(result)
    return result


def copy_results(src_dir, dst_dir, suffixes):
    """Copy all files from src_dir to dst_dir that match any of the given suffixes"""
    cmd = ["rsync", "-avz", "--include=*/"]
    for s in suffixes:
        pattern = f"--include=*{s}"
        cmd.append(pattern)
    cmd.append("--exclude=*")
    cmd.append(src_dir.rstrip('/') + '/')
    cmd.append(dst_dir)
    run(cmd, check=True)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def to_float(x):
    try:
        return x.float()
    except AttributeError:
        return x


def disable_keypoint_loss(w2c, b):
    object_front_normal = repeat(torch.tensor([1., 0., 0.], device=w2c.device), "c -> b c 1", b=b)
    R_world_to_cam = w2c[:, :3, :3]
    cam_forward_in_world = R_world_to_cam.transpose(1, 2) @ repeat(torch.tensor([0., 0., 1.], device=w2c.device),
                                                                   "c -> b c 1", b=b)
    similarity = F.cosine_similarity(cam_forward_in_world, object_front_normal).abs()
    keypoint_gt_flag = (similarity > 0.25).squeeze()
    return keypoint_gt_flag


def draw_keypoints(image_t, keypoint, gt_flag=None, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    b, f = image_t.shape[:2]
    img_size = image_t.shape[-1]
    image_t = rearrange(image_t, "b f ... -> (b f) ...")
    keypoint = rearrange((keypoint + 1) / 2 * img_size, "b f ... -> (b f) ...")
    if gt_flag is None:
        gt_flag = torch.ones(b*f, device=image_t.device, dtype=torch.bool)
    elif gt_flag.dim() == 2:
        gt_flag = rearrange(gt_flag, "b f -> (b f)")
    out_list = []
    for img, k, flag in zip(image_t, keypoint, gt_flag):
        if flag:
            img_np = (img.clone() * 255).permute(1, 2, 0).clamp(0, 255).contiguous().cpu().numpy().astype(np.uint8)
            for i, (x, y) in enumerate(k[:, :2]):
                x, y = int(x.item()), int(y.item())
                cv2.circle(img_np, center=(x, y), radius=radius, color=circle_color, thickness=-1)
                cv2.putText(
                    img_np, text=str(i), org=(x + radius + 1, y - radius - 1),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=text_color, thickness=1, lineType=cv2.LINE_AA
                )
            out_list.append(torch.from_numpy(img_np).permute(2, 0, 1) / 255)
        else:
            out_list.append(img.cpu())
    out_t = rearrange(torch.stack(out_list), "(b f) ... -> b f ...", b=b, f=f)
    return out_t


def mask_diff(mask1, mask2):
    """Return difference of two masks, where pixels are 1 if different, 0 if same"""
    assert mask1.shape == mask2.shape
    if isinstance(mask1, torch.Tensor):
        return (mask1 - mask2).abs().clamp(0, 1)
    elif isinstance(mask1, np.ndarray):
        return np.abs(mask1 - mask2).clip(0, 1)
    else:
        raise NotImplementedError


def remove_background(image, bg_value=0, c_dim=-3):
    """Add alpha channel to image where background pixels are transparent"""
    if isinstance(image, torch.Tensor):
        if image.shape[c_dim] == 1:
            image = torch.cat([image, image, image], dim=c_dim)
        bg = torch.all(image == bg_value, dim=c_dim, keepdim=True)
        alpha = (~bg).float()
        image = torch.cat([image, alpha], dim=c_dim)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError
    else:
        raise NotImplementedError
    return image


