#!/usr/bin/env python3
"""
Mouse Training Pipeline
=======================
Complete pipeline for training, inference, mesh extraction, and HTML report generation.

Usage:
    # Full pipeline (debug mode)
    python scripts/run_mouse_pipeline.py --mode debug

    # Full pipeline (full training)
    python scripts/run_mouse_pipeline.py --mode full

    # Inference only (from existing checkpoint)
    python scripts/run_mouse_pipeline.py --mode inference --checkpoint results/mouse/latest.pth

    # Generate report only
    python scripts/run_mouse_pipeline.py --mode report --results_dir results/mouse
"""

import os
import sys
import argparse
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import glob

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MousePipeline:
    """Complete pipeline for mouse 3D reconstruction training and evaluation."""

    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = args.exp_name or f"mouse_{args.mode}_{self.timestamp}"
        self.results_dir = Path(args.results_dir) / self.exp_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.state = {
            'train_complete': False,
            'inference_complete': False,
            'mesh_complete': False,
            'report_complete': False,
            'checkpoint_path': args.checkpoint,
            'errors': []
        }

        # Logging
        self.log_file = self.results_dir / "pipeline.log"

    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")

    def run_command(self, cmd: List[str], desc: str) -> bool:
        """Run a shell command and return success status."""
        self.log(f"Running: {desc}")
        self.log(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=self.args.timeout
            )

            if result.returncode == 0:
                self.log(f"SUCCESS: {desc}")
                return True
            else:
                self.log(f"FAILED: {desc}", "ERROR")
                self.log(f"stderr: {result.stderr[:1000]}", "ERROR")
                self.state['errors'].append({
                    'step': desc,
                    'error': result.stderr[:500]
                })
                return False

        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: {desc}", "ERROR")
            return False
        except Exception as e:
            self.log(f"EXCEPTION: {desc} - {str(e)}", "ERROR")
            return False

    def step_train(self) -> bool:
        """Step 1: Training."""
        self.log("=" * 60)
        self.log("STEP 1: Training")
        self.log("=" * 60)

        if self.args.skip_train:
            self.log("Skipping training (--skip_train)")
            return True

        # Determine iterations
        num_iters = 5000 if self.args.mode == 'debug' else 200000
        if self.args.num_iters:
            num_iters = self.args.num_iters

        cmd = [
            "python", "run.py",
            "--config-name", "train_mouse_debug",
            f"hydra.run.dir=.",
            f"exp_name={self.exp_name}",
            f"output_dir={self.results_dir}",
            f"num_iters={num_iters}",
            f"save_checkpoint_freq={self.args.save_freq}",
            f"log_image_freq={self.args.log_freq}",
        ]

        # Data directory
        if self.args.data_dir:
            cmd.append(f"dataset.train_data_dir={self.args.data_dir}")
            cmd.append(f"dataset.val_data_dir={self.args.data_dir}")

        # Resume from checkpoint
        if self.args.checkpoint:
            cmd.append(f"resume={self.args.checkpoint}")
        else:
            cmd.append("resume=null")

        success = self.run_command(cmd, "Training")

        if success:
            # Find latest checkpoint
            checkpoints = sorted(glob.glob(str(self.results_dir / "*.pth")))
            if checkpoints:
                self.state['checkpoint_path'] = checkpoints[-1]
                self.log(f"Latest checkpoint: {self.state['checkpoint_path']}")
            self.state['train_complete'] = True

        return success

    def step_inference(self) -> bool:
        """Step 2: Inference and visualization."""
        self.log("=" * 60)
        self.log("STEP 2: Inference & Visualization")
        self.log("=" * 60)

        if not self.state['checkpoint_path']:
            self.log("No checkpoint found, skipping inference", "WARN")
            return False

        vis_dir = self.results_dir / "visualization"
        vis_dir.mkdir(exist_ok=True)

        # Determine test data directory
        test_data_dir = self.args.test_data_dir or self.args.data_dir
        if not test_data_dir:
            self.log("No test data directory specified", "WARN")
            return False

        cmd = [
            "python", "visualization/visualize_results_fauna.py",
            "--config-name", "test_fauna",
            f"hydra.run.dir=.",
            f"checkpoint_path={self.state['checkpoint_path']}",
            f"output_dir={vis_dir}",
            f"dataset.test_data_dir={test_data_dir}",
            "render_modes=[input_view,other_views,rotation]",
        ]

        success = self.run_command(cmd, "Inference & Visualization")

        if success:
            self.state['inference_complete'] = True

        return success

    def step_mesh_extraction(self) -> bool:
        """Step 3: Mesh extraction."""
        self.log("=" * 60)
        self.log("STEP 3: Mesh Extraction")
        self.log("=" * 60)

        if not self.state['checkpoint_path']:
            self.log("No checkpoint found, skipping mesh extraction", "WARN")
            return False

        mesh_dir = self.results_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True)

        # Create mesh extraction script inline
        mesh_script = self.results_dir / "extract_mesh.py"
        mesh_script_content = f'''
import sys
sys.path.insert(0, "{PROJECT_ROOT}")

import torch
import numpy as np
from pathlib import Path
from model import build_model
from omegaconf import OmegaConf
import glob
from PIL import Image
import torchvision.transforms as T

def extract_meshes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint_path = "{self.state['checkpoint_path']}"
    cp = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use default
    if 'cfg' in cp:
        cfg = OmegaConf.create(cp['cfg'])
    else:
        cfg = OmegaConf.load("{PROJECT_ROOT}/config/model/mouse.yaml")

    # Build model
    model = build_model(cfg)
    model.load_model_state(cp)
    model.to(device)
    model.eval()

    # Find test images
    test_dir = "{self.args.test_data_dir or self.args.data_dir}"
    if test_dir:
        image_paths = sorted(Path(test_dir).rglob("*_rgb.png"))[:5]  # First 5 images
    else:
        print("No test directory specified")
        return

    mesh_dir = Path("{mesh_dir}")

    for img_path in image_paths:
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            img_tensor = T.ToTensor()(img).unsqueeze(0).unsqueeze(0).to(device)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.squeeze(1), size=(256, 256), mode='bilinear'
            ).unsqueeze(1)

            with torch.no_grad():
                # Get shape prediction
                prior_shape, dino_pred, bank_embedding = model.netBase(
                    total_iter=999999, is_training=False,
                    batch=[img_tensor], bank_enc=model.netInstance.netEncoder
                )
                shape, pose_raw, pose, mvp, w2c, campos, texture_pred, im_features, deform, all_arti_params, light, forward_aux = \\
                    model.netInstance(img_tensor, prior_shape, 999, 999999, is_training=False)

                # Export mesh
                basename = img_path.stem.replace("_rgb", "")
                mesh_path = mesh_dir / f"{{basename}}_mesh.obj"

                v_pos = shape.v_pos[0].cpu().numpy()
                t_pos_idx = shape.t_pos_idx[0].cpu().numpy()

                with open(mesh_path, 'w') as f:
                    for v in v_pos:
                        f.write(f"v {{v[0]}} {{v[1]}} {{v[2]}}\\n")
                    for face in t_pos_idx:
                        f.write(f"f {{face[0]+1}} {{face[1]+1}} {{face[2]+1}}\\n")

                print(f"Saved: {{mesh_path}}")

        except Exception as e:
            print(f"Error processing {{img_path}}: {{e}}")

if __name__ == "__main__":
    extract_meshes()
'''

        with open(mesh_script, 'w') as f:
            f.write(mesh_script_content)

        cmd = ["python", str(mesh_script)]
        success = self.run_command(cmd, "Mesh Extraction")

        if success:
            self.state['mesh_complete'] = True

        return success

    def step_generate_report(self) -> bool:
        """Step 4: Generate HTML report."""
        self.log("=" * 60)
        self.log("STEP 4: Generate HTML Report")
        self.log("=" * 60)

        report_path = self.results_dir / "report.html"

        # Collect results
        vis_dir = self.results_dir / "visualization"
        mesh_dir = self.results_dir / "meshes"
        images_dir = self.results_dir / "images"

        # Find visualization images
        input_views = sorted(glob.glob(str(vis_dir / "*_input_view*.png")))
        other_views = sorted(glob.glob(str(vis_dir / "*_other_view*.png")))
        rotation_videos = sorted(glob.glob(str(vis_dir / "*_rotation*.mp4")))
        meshes = sorted(glob.glob(str(mesh_dir / "*.obj")))
        training_images = sorted(glob.glob(str(images_dir / "*.png")))[-10:]  # Last 10

        # Generate HTML
        html_content = self._generate_html(
            input_views=input_views,
            other_views=other_views,
            rotation_videos=rotation_videos,
            meshes=meshes,
            training_images=training_images
        )

        with open(report_path, 'w') as f:
            f.write(html_content)

        self.log(f"Report saved: {report_path}")
        self.state['report_complete'] = True

        return True

    def _generate_html(self, input_views, other_views, rotation_videos, meshes, training_images) -> str:
        """Generate HTML report content."""

        def img_to_base64_tag(img_path, max_width=300):
            """Convert image to inline HTML img tag."""
            rel_path = os.path.relpath(img_path, self.results_dir)
            return f'<img src="{rel_path}" style="max-width: {max_width}px; margin: 5px;">'

        def video_tag(video_path, max_width=400):
            """Generate video tag."""
            rel_path = os.path.relpath(video_path, self.results_dir)
            return f'''<video width="{max_width}" controls>
                <source src="{rel_path}" type="video/mp4">
            </video>'''

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mouse 3D Reconstruction Report - {self.exp_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
            border-left: 4px solid #4CAF50;
            padding-left: 15px;
        }}
        .meta-info {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .meta-info p {{
            margin: 5px 0;
        }}
        .gallery {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }}
        .gallery img {{
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .video-container {{
            margin: 20px 0;
        }}
        .video-container video {{
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .status {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-success {{
            background: #4CAF50;
            color: white;
        }}
        .status-failed {{
            background: #f44336;
            color: white;
        }}
        .status-pending {{
            background: #ff9800;
            color: white;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f5f5f5;
        }}
        .mesh-list {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }}
        .mesh-list li {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Mouse 3D Reconstruction Report</h1>

        <div class="meta-info">
            <p><strong>Experiment:</strong> {self.exp_name}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Mode:</strong> {self.args.mode}</p>
            <p><strong>Checkpoint:</strong> {self.state['checkpoint_path'] or 'N/A'}</p>
        </div>

        <h2>Pipeline Status</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Training</td>
                <td><span class="status {'status-success' if self.state['train_complete'] else 'status-pending'}">
                    {'Complete' if self.state['train_complete'] else 'Pending/Skipped'}
                </span></td>
            </tr>
            <tr>
                <td>Inference</td>
                <td><span class="status {'status-success' if self.state['inference_complete'] else 'status-pending'}">
                    {'Complete' if self.state['inference_complete'] else 'Pending'}
                </span></td>
            </tr>
            <tr>
                <td>Mesh Extraction</td>
                <td><span class="status {'status-success' if self.state['mesh_complete'] else 'status-pending'}">
                    {'Complete' if self.state['mesh_complete'] else 'Pending'}
                </span></td>
            </tr>
        </table>
'''

        # Training Progress
        if training_images:
            html += '''
        <h2>Training Progress</h2>
        <p>Last 10 training visualizations:</p>
        <div class="gallery">
'''
            for img in training_images:
                html += f'            {img_to_base64_tag(img, 250)}\n'
            html += '        </div>\n'

        # Input View Reconstruction
        if input_views:
            html += '''
        <h2>Input View Reconstruction</h2>
        <div class="gallery">
'''
            for img in input_views[:12]:
                html += f'            {img_to_base64_tag(img, 300)}\n'
            html += '        </div>\n'

        # Novel Views
        if other_views:
            html += '''
        <h2>Novel View Synthesis</h2>
        <div class="gallery">
'''
            for img in other_views[:24]:
                html += f'            {img_to_base64_tag(img, 200)}\n'
            html += '        </div>\n'

        # Rotation Videos
        if rotation_videos:
            html += '''
        <h2>360° Rotation Videos</h2>
        <div class="video-container">
'''
            for video in rotation_videos[:4]:
                html += f'            {video_tag(video, 400)}\n'
            html += '        </div>\n'

        # Meshes
        if meshes:
            html += '''
        <h2>Extracted 3D Meshes</h2>
        <div class="mesh-list">
            <ul>
'''
            for mesh in meshes:
                rel_path = os.path.relpath(mesh, self.results_dir)
                html += f'                <li><a href="{rel_path}">{os.path.basename(mesh)}</a></li>\n'
            html += '''            </ul>
        </div>
'''

        # Errors
        if self.state['errors']:
            html += '''
        <h2>Errors</h2>
        <div style="background: #ffebee; padding: 15px; border-radius: 5px;">
'''
            for err in self.state['errors']:
                html += f'            <p><strong>{err["step"]}:</strong> {err["error"][:200]}</p>\n'
            html += '        </div>\n'

        html += '''
    </div>
</body>
</html>
'''
        return html

    def run(self):
        """Run the complete pipeline."""
        self.log("=" * 60)
        self.log(f"Starting Mouse Pipeline: {self.exp_name}")
        self.log(f"Mode: {self.args.mode}")
        self.log(f"Results: {self.results_dir}")
        self.log("=" * 60)

        steps = []

        if self.args.mode in ['debug', 'full', 'train']:
            steps.append(('train', self.step_train))

        if self.args.mode in ['debug', 'full', 'inference', 'train']:
            steps.append(('inference', self.step_inference))
            steps.append(('mesh', self.step_mesh_extraction))

        # Always generate report
        steps.append(('report', self.step_generate_report))

        for step_name, step_fn in steps:
            self.log(f"\n>>> Starting step: {step_name}")
            success = step_fn()
            if not success and step_name in ['train']:
                self.log(f"Critical step '{step_name}' failed, stopping pipeline", "ERROR")
                break

        # Final summary
        self.log("\n" + "=" * 60)
        self.log("PIPELINE COMPLETE")
        self.log("=" * 60)
        self.log(f"Results directory: {self.results_dir}")
        self.log(f"Report: {self.results_dir / 'report.html'}")

        # Save state
        state_path = self.results_dir / "pipeline_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Mouse 3D Reconstruction Pipeline")

    # Mode
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'full', 'train', 'inference', 'report'],
                        help='Pipeline mode')

    # Paths
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not specified)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--data_dir', type=str,
                        default='/home/joon/data/fauna_mouse_only',
                        help='Training data directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Test data directory (defaults to data_dir)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for resume/inference')

    # Training
    parser.add_argument('--num_iters', type=int, default=None,
                        help='Number of training iterations')
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='Checkpoint save frequency')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Image logging frequency')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training step')

    # Misc
    parser.add_argument('--timeout', type=int, default=86400,
                        help='Timeout per step in seconds (default: 24h)')

    args = parser.parse_args()

    pipeline = MousePipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
