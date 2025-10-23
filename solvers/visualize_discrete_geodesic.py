# pytorch >= 2.0
import torch
from torch import nn
from typing import Callable, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from pathlib import Path

# Import the utilities from discrete_geodesic.py
from discrete_geodesic import (
    jacobian, tangent_project_matrix, project_to_manifold,
    batch_project_to_manifold, path_length_huber
)


def optimize_path_on_implicit_manifold_visualized(
    g: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    t: torch.Tensor,
    L: int,
    init_Y: torch.Tensor = None,
    steps: int = 1000,
    lr: float = 1e-1,
    eps_length: float = 1e-3,
    project_every: int = 5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    max_newton_projection: int = 5,
    step_clip: float = 0.25,
    save_every: int = 10,
) -> Tuple[torch.Tensor, dict, List[np.ndarray]]:
    """
    Modified version that saves snapshots of the path during optimization.
    Returns the optimized path, history, and list of path snapshots.
    """
    d = s.numel()
    s = s.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)

    # Initialize path
    if init_Y is None:
        alphas = torch.linspace(0., 1., L+1, device=device, dtype=dtype).unsqueeze(-1)
        Y0 = (1 - alphas) * s.unsqueeze(0) + alphas * t.unsqueeze(0)
        fixed = torch.zeros(L+1, dtype=torch.bool, device=device)
        fixed[0] = True
        fixed[-1] = True
        Y0 = batch_project_to_manifold(Y0, g, fixed_mask=fixed, max_newton=max_newton_projection)
    else:
        assert init_Y.shape == (L+1, d)
        Y0 = init_Y.to(device=device, dtype=dtype)

    Y = nn.Parameter(Y0.clone())
    fixed_mask = torch.zeros(L+1, dtype=torch.bool, device=device)
    fixed_mask[0] = True
    fixed_mask[-1] = True

    opt = torch.optim.SGD([Y], lr=lr, momentum=0.0)

    history = {"loss": [], "step": []}
    snapshots = []

    # Save initial state
    snapshots.append(Y.detach().cpu().numpy().copy())

    for k in range(steps):
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            Y.data[0] = s
            Y.data[-1] = t

        loss = path_length_huber(Y, eps=eps_length)
        loss.backward()

        with torch.no_grad():
            for i in range(1, L):
                Ji = jacobian(g, Y.data[i])
                Pi = tangent_project_matrix(Ji)
                gi = Y.grad[i] if Y.grad[i] is not None else torch.zeros_like(Y.data[i])
                gi_proj = Pi @ gi
                nrm = torch.linalg.norm(gi_proj)
                if nrm > step_clip:
                    gi_proj = gi_proj * (step_clip / (nrm + 1e-12))
                Y.grad[i].copy_(gi_proj)

            Y.grad[0].zero_()
            Y.grad[-1].zero_()

        opt.step()

        if (k+1) % project_every == 0:
            with torch.no_grad():
                Y.data = batch_project_to_manifold(
                    Y.data, g, fixed_mask=fixed_mask, max_newton=max_newton_projection
                )
                Y.data[0] = s
                Y.data[-1] = t

        history["loss"].append(float(loss.detach().cpu()))
        history["step"].append(k)

        # Save snapshots
        if k % save_every == 0 or k == steps - 1:
            snapshots.append(Y.detach().cpu().numpy().copy())

        if (k+1) % (steps // 3 if steps >= 3 else 1) == 0:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    # Final projection
    with torch.no_grad():
        Y.data = batch_project_to_manifold(
            Y.data, g, fixed_mask=fixed_mask, max_newton=max_newton_projection
        )
        Y.data[0] = s
        Y.data[-1] = t

    snapshots.append(Y.detach().cpu().numpy().copy())

    return Y.detach(), history, snapshots


def create_sphere_mesh(radius=1.0, resolution=30):
    """Create sphere mesh for visualization"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_visualization_video(
    snapshots: List[np.ndarray],
    history: dict,
    s: np.ndarray,
    t: np.ndarray,
    output_path: str = "geodesic_optimization.mp4",
    fps: int = 30,
    save_every: int = 10,
):
    """
    Create an animated video showing the optimization process.
    """
    fig = plt.figure(figsize=(16, 6))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')

    # Loss plot
    ax2 = fig.add_subplot(122)

    # Create sphere mesh
    x_sphere, y_sphere, z_sphere = create_sphere_mesh()

    # Plot setup for 3D
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color='cyan',
                     edgecolor='none', shade=True)

    # Plot start and end points
    ax1.scatter(*s, color='green', s=200, marker='o', label='Start',
                edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(*t, color='red', s=200, marker='o', label='End',
                edgecolors='black', linewidths=2, zorder=10)

    # Initialize path line
    line, = ax1.plot([], [], [], 'b-o', linewidth=2, markersize=4,
                     label='Path', alpha=0.8)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Geodesic Path Optimization on Sphere')
    ax1.legend()
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])

    # Set equal aspect ratio
    ax1.set_box_aspect([1, 1, 1])

    # Plot setup for loss
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('Path Length (Loss)')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    loss_line, = ax2.plot([], [], 'b-', linewidth=2)
    current_point, = ax2.plot([], [], 'ro', markersize=8)

    # Text annotations
    step_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes,
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    loss_text = ax1.text2D(0.02, 0.90, '', transform=ax1.transAxes,
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        loss_line.set_data([], [])
        current_point.set_data([], [])
        step_text.set_text('')
        loss_text.set_text('')
        return line, loss_line, current_point, step_text, loss_text

    def animate(frame):
        # Update 3D path
        path = snapshots[frame]
        line.set_data(path[:, 0], path[:, 1])
        line.set_3d_properties(path[:, 2])

        # Update loss plot
        actual_step = frame * save_every
        step_idx = min(actual_step, len(history['loss']) - 1)

        steps_to_plot = history['step'][:step_idx+1]
        losses_to_plot = history['loss'][:step_idx+1]

        loss_line.set_data(steps_to_plot, losses_to_plot)
        if len(steps_to_plot) > 0:
            current_point.set_data([steps_to_plot[-1]], [losses_to_plot[-1]])
            ax2.set_xlim([0, max(history['step'])])
            ax2.set_ylim([min(history['loss']) * 0.95, max(history['loss']) * 1.05])

        # Update text
        step_text.set_text(f'Step: {actual_step}')
        if len(losses_to_plot) > 0:
            loss_text.set_text(f'Loss: {losses_to_plot[-1]:.6f}')

        # Rotate view slightly for better 3D perception
        ax1.view_init(elev=20, azim=frame * 0.5)

        return line, loss_line, current_point, step_text, loss_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(snapshots), interval=1000/fps,
        blit=False, repeat=True
    )

    # Save animation
    print(f"Saving animation to {output_path}...")

    # Try different writers
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(output_path, writer=writer)
        print(f"Animation saved successfully using FFMpegWriter!")
    except Exception as e:
        print(f"FFMpegWriter failed: {e}")
        try:
            # Try pillow writer for gif
            output_path = str(output_path).replace('.mp4', '.gif')
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
            print(f"Animation saved as GIF successfully!")
        except Exception as e2:
            print(f"PillowWriter also failed: {e2}")
            print("Available writers:", animation.writers.list())
            # Save frames as individual images
            frames_dir = Path(output_path).parent / "frames"
            frames_dir.mkdir(exist_ok=True)
            print(f"Saving individual frames to {frames_dir}...")
            for i in range(len(snapshots)):
                animate(i)
                plt.savefig(frames_dir / f"frame_{i:04d}.png", dpi=100, bbox_inches='tight')
            print(f"Frames saved! You can create a video with: ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")

    plt.close()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Implicit manifold: unit sphere in R^3
    def g_sphere(y: torch.Tensor) -> torch.Tensor:
        result = y.dot(y) - 1.0
        return result.view(1)  
    d = 3
    s = torch.tensor([1., 0., 0.])
    t = torch.tensor([0., 1., 0.])

    print("Running optimization with visualization...")
    Y_opt, hist, snapshots = optimize_path_on_implicit_manifold_visualized(
        g=g_sphere,
        s=s, t=t,
        L=40,
        steps=800,
        lr=0.2,
        eps_length=1e-3,
        project_every=5,
        device="cpu",
        dtype=torch.float64,
        max_newton_projection=8,
        step_clip=0.1,
        save_every=5,  # Save every 5 steps
    )

    print(f"Optimization complete!")
    print(f"Final loss (approx length): {hist['loss'][-1]}")
    print(f"Number of snapshots captured: {len(snapshots)}")

    # Create output directory if needed
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "geodesic_optimization.mp4"

    # Create visualization
    print("\nCreating visualization video...")
    create_visualization_video(
        snapshots=snapshots,
        history=hist,
        s=s.numpy(),
        t=t.numpy(),
        output_path=str(output_path),
        fps=30,
        save_every=5,
    )

    print(f"\nVisualization saved to: {output_path}")
