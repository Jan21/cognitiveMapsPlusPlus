"""
Animate the evolution of node embeddings across training checkpoints.

This script loads multiple checkpoints from a training run, extracts node embeddings,
applies UMAP dimensionality reduction, aligns the embeddings using Orthogonal Procrustes,
and creates an animated GIF showing how the embeddings evolve during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
import sys
from typing import List, Tuple
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Install with: pip install Pillow")

from scipy.linalg import orthogonal_procrustes
from model.lightningmodule import PathPredictionModule



def find_checkpoints(checkpoint_dir: str, exclude_last: bool = True) -> List[Path]:
    """
    Find all checkpoint files in a directory, sorted by epoch number.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        exclude_last: Whether to exclude 'last.ckpt' file

    Returns:
        List of checkpoint paths sorted by epoch
    """
    checkpoint_path = Path(checkpoint_dir)

    # Find all .ckpt files
    checkpoint_files = list(checkpoint_path.glob("*.ckpt"))

    # Exclude last.ckpt if requested
    if exclude_last:
        checkpoint_files = [f for f in checkpoint_files if f.name != "last.ckpt"]

    # Sort by epoch number extracted from filename
    def extract_epoch(filepath: Path) -> int:
        """Extract epoch number from checkpoint filename."""
        match = re.search(r'epoch[=_](\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return -1

    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)

    print(f"Found {len(checkpoint_files)} checkpoints")
    for ckpt in checkpoint_files:
        print(f"  - {ckpt.name}")

    return checkpoint_files


def load_embeddings_from_checkpoint(checkpoint_path: Path) -> np.ndarray:
    """
    Load node embeddings from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Numpy array of embeddings [vocab_size, d_model]
    """
    model = PathPredictionModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    with torch.no_grad():
        embeddings = model.model.node_embedding.weight.cpu().numpy()

    # Exclude padding token (last embedding)
    return embeddings[:-1, :]


def apply_umap(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction with fixed hyperparameters.

    Args:
        embeddings: High-dimensional embeddings [num_nodes, d_model]
        n_components: Number of output dimensions (default: 3)

    Returns:
        Reduced embeddings [num_nodes, n_components]
    """
    reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=2.0,
        spread=2.0,
        n_components=n_components,
        metric='cosine',
        random_state=42
    )
    return reducer.fit_transform(embeddings)


def align_embeddings_procrustes(embeddings_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Align a list of embedding matrices using Orthogonal Procrustes with centering.

    Each embedding matrix is aligned to the previous one, creating a smooth
    transition across the sequence. The embeddings are centered before alignment
    to prevent drift, then transformed to align their orientation.

    Args:
        embeddings_list: List of embedding matrices [num_checkpoints, num_nodes, n_dims]

    Returns:
        List of aligned embedding matrices
    """
    if len(embeddings_list) <= 1:
        return embeddings_list

    # Center the first embedding and use it as reference
    reference = embeddings_list[0]
    reference_centered = reference - reference.mean(axis=0)

    aligned = [reference_centered]

    for i in range(1, len(embeddings_list)):
        # Center the current embedding
        current = embeddings_list[i]
        current_mean = current.mean(axis=0)
        current_centered = current - current_mean

        # Center the previous aligned embedding (for Procrustes)
        prev_aligned = aligned[i-1]
        prev_mean = prev_aligned.mean(axis=0)
        prev_centered = prev_aligned - prev_mean

        # Find rotation matrix R such that current_centered @ R ≈ prev_centered
        R, scale = orthogonal_procrustes(current_centered, prev_centered)

        # Apply rotation and translate to match previous centroid
        aligned_embedding = current_centered @ R + prev_mean
        aligned.append(aligned_embedding)

    return aligned


def create_3d_frame(embedding_3d: np.ndarray, epoch: int, frame_index: int,
                    total_frames: int, title_prefix: str = "",
                    rotation_speed: float = 1.0) -> plt.Figure:
    """
    Create a single 3D visualization frame for the animation with rotating view.

    Args:
        embedding_3d: 3D embedding coordinates [num_nodes, 3]
        epoch: Epoch number for title
        frame_index: Index of current frame (0 to total_frames-1)
        total_frames: Total number of frames in animation
        title_prefix: Optional prefix for title
        rotation_speed: Speed multiplier for rotation (1.0 = full rotation)

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = np.arange(len(embedding_3d))
    scatter = ax.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=colors,
        cmap='viridis',
        alpha=0.7,
        s=50
    )

    title = f'{title_prefix}UMAP Embedding - Epoch {epoch}'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Remove axis labels and ticks for cleaner visualization
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide axis panes and grid lines
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

    # Hide axis lines completely
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Make the axes completely invisible
    ax.set_axis_off()

    # Rotate the camera view
    # azimuth: rotation around z-axis (horizontal rotation)
    # elevation: angle above the xy-plane (vertical tilt)
    progress = frame_index / max(total_frames - 1, 1)
    azimuth = progress * 360 * rotation_speed  # Full rotation
    elevation = 2  # Keep elevation constant for stable viewing

    ax.view_init(elev=elevation, azim=azimuth)

    # Set consistent axis limits across all frames for stability
    # These will be computed globally before creating frames

    # Minimize margins to reduce white space
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0, hspace=0)

    return fig


def compute_global_limits(aligned_embeddings: List[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute global axis limits across all embeddings for consistent visualization.

    Args:
        aligned_embeddings: List of aligned embedding matrices

    Returns:
        Tuple of (xlim, ylim, zlim) each as (min, max)
    """
    all_embeddings = np.concatenate(aligned_embeddings, axis=0)

    margin = 0.1  # 10% margin

    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    z_min, z_max = all_embeddings[:, 2].min(), all_embeddings[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    xlim = (x_min - margin * x_range, x_max + margin * x_range)
    ylim = (y_min - margin * y_range, y_max + margin * y_range)
    zlim = (z_min - margin * z_range, z_max + margin * z_range)

    return xlim, ylim, zlim


def create_animation(
    checkpoint_dir: str,
    output_path: str = "visualize/embedding_evolution.gif",
    fps: int = 2,
    title_prefix: str = "",
    rotation_speed: float = 1.0
):
    """
    Create an animated GIF showing embedding evolution across checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        output_path: Path to save the output GIF
        fps: Frames per second for the animation
        title_prefix: Optional prefix for frame titles
        rotation_speed: Camera rotation speed (1.0 = one full rotation, 0.0 = no rotation)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")

    if not PIL_AVAILABLE:
        raise ImportError("PIL not installed. Install with: pip install Pillow")

    # Find all checkpoints
    checkpoint_files = find_checkpoints(checkpoint_dir, exclude_last=True)

    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    print("\nLoading embeddings from checkpoints...")
    embeddings_list = []
    epoch_numbers = []

    for ckpt_path in checkpoint_files:
        print(f"Loading {ckpt_path.name}...")
        embeddings = load_embeddings_from_checkpoint(ckpt_path)
        embeddings_list.append(embeddings)

        # Extract epoch number
        match = re.search(r'epoch[=_](\d+)', ckpt_path.name)
        epoch_num = int(match.group(1)) if match else -1
        epoch_numbers.append(epoch_num)

    print(f"\nLoaded {len(embeddings_list)} embedding matrices")
    print(f"Embedding shape: {embeddings_list[0].shape}")

    # Apply UMAP to all embeddings
    print("\nApplying UMAP dimensionality reduction...")
    reduced_embeddings = []
    for i, embeddings in enumerate(embeddings_list):
        print(f"Processing checkpoint {i+1}/{len(embeddings_list)}...")
        reduced = apply_umap(embeddings, n_components=3)
        reduced_embeddings.append(reduced)

    print("\nAligning embeddings using Orthogonal Procrustes...")
    aligned_embeddings = align_embeddings_procrustes(reduced_embeddings)

    # Compute global axis limits
    xlim, ylim, zlim = compute_global_limits(aligned_embeddings)
    print(f"Global limits - X: {xlim}, Y: {ylim}, Z: {zlim}")

    # Create frames
    print("\nGenerating animation frames...")
    frames = []
    total_frames = len(aligned_embeddings)

    for i, (embedding, epoch) in enumerate(zip(aligned_embeddings, epoch_numbers)):
        print(f"Creating frame {i+1}/{total_frames}...")

        fig = create_3d_frame(
            embedding, epoch, i, total_frames,
            title_prefix, rotation_speed
        )
        ax = fig.axes[0]

        # Set consistent axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # Convert figure to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(Image.fromarray(image))

        plt.close(fig)

    # Save as GIF
    print(f"\nSaving animation to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = int(1000 / fps)  # Duration in milliseconds
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )

    print(f"Animation saved successfully!")
    print(f"Total frames: {len(frames)}")
    print(f"Duration per frame: {duration}ms ({fps} FPS)")


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF of embedding evolution across training checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/home/jan/projects/CIIRC/cognitiveMapsPlusPlus/temp/checkpoints/PathPrediction_graph_torus_model_model.architecture.diffusion_upsample.Diffusion_ResidualUpsample_trial0",
        help="Directory containing checkpoint files (e.g., temp/checkpoints/run_name/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualize/embedding_evolution.gif",
        help="Path to save the output GIF"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for the animation (default: 2)"
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional prefix for frame titles"
    )
    parser.add_argument(
        "--rotation-speed",
        type=float,
        default=1.0,
        help="Camera rotation speed (1.0 = one full 360° rotation, 0.0 = no rotation, 2.0 = two rotations)"
    )

    args = parser.parse_args()

    create_animation(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        fps=args.fps,
        title_prefix=args.title_prefix,
        rotation_speed=args.rotation_speed
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
