# pytorch >= 2.0
import torch
from torch import nn
from typing import Callable, Tuple

# ---------- Utilities for implicit manifolds g(y)=0 ----------

def jacobian(g: Callable[[torch.Tensor], torch.Tensor],
             y: torch.Tensor) -> torch.Tensor:
    """
    Compute J_g(y) with shape (m, d) for a single point y (d,)
    where g: R^d -> R^m. Uses autograd jacobian.
    """
    y = y.detach().requires_grad_(True)
    # torch.autograd.functional.jacobian expects function -> tensor
    def g_single(z):
        return g(z)
    J = torch.autograd.functional.jacobian(g_single, y, create_graph=False)
    # J has shape (m,d)
    return J.detach()


def tangent_project_matrix(J: torch.Tensor) -> torch.Tensor:
    """
    Return the tangent projector P = I - J^T (J J^T)^(-1) J.
    J: (m, d), full row-rank assumed near the manifold.
    """
    d = J.shape[1]
    JJt = J @ J.T                      # (m,m)
    # Add tiny Tikhonov for stability
    reg = 1e-10 * torch.eye(JJt.shape[0], dtype=JJt.dtype, device=JJt.device)
    JJt = JJt + reg
    invJJt = torch.linalg.solve(JJt, torch.eye(JJt.shape[0], device=JJt.device, dtype=JJt.dtype))
    P = torch.eye(d, device=J.device, dtype=J.dtype) - J.T @ invJJt @ J
    return P


@torch.no_grad()
def project_to_manifold(
    z: torch.Tensor,
    g: Callable[[torch.Tensor], torch.Tensor],
    max_newton: int = 5,
    tol: float = 1e-9
) -> torch.Tensor:
    """
    Orthogonal projection of a single point z (d,) onto {y | g(y)=0}
    using Newton–Gauss correction:
        y_{k+1} = y_k - J^T (J J^T)^(-1) g(y_k)
    """
    y = z.clone()
    for _ in range(max_newton):
        val = g(y)            # (m,)
        nrm = torch.linalg.norm(val)
        if nrm < tol:
            break
        J = jacobian(g, y)    # (m,d)
        JJt = J @ J.T
        # Stabilize solve
        reg = 1e-12 * torch.eye(JJt.shape[0], device=y.device, dtype=y.dtype)
        step = J.T @ torch.linalg.solve(JJt + reg, val)
        y = y - step
    return y


@torch.no_grad()
def batch_project_to_manifold(
    Z: torch.Tensor,
    g: Callable[[torch.Tensor], torch.Tensor],
    fixed_mask: torch.Tensor,
    max_newton: int = 5
) -> torch.Tensor:
    """
    Project all rows of Z (N,d) independently, skipping rows where fixed_mask[i]=True
    """
    Y = Z.clone()
    for i in range(Y.shape[0]):
        if not bool(fixed_mask[i].item()):
            Y[i] = project_to_manifold(Y[i], g, max_newton=max_newton)
    return Y


def path_length_huber(Y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Smooth (Huberized) discrete length:
        sum_i sqrt(||Y[i+1]-Y[i]||^2 + eps^2)
    Y: (L+1, d)
    """
    deltas = Y[1:] - Y[:-1]
    seg = torch.sqrt(torch.sum(deltas * deltas, dim=1) + eps * eps)
    return seg.sum()


# ---------- Optimizer ----------

def optimize_path_on_implicit_manifold(
    g: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,               # (d,)
    t: torch.Tensor,               # (d,)
    L: int,                        # number of segments; path has L+1 points
    init_Y: torch.Tensor = None,   # optional (L+1,d) initialization
    steps: int = 1000,
    lr: float = 1e-1,
    eps_length: float = 1e-3,
    project_every: int = 5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    max_newton_projection: int = 5,
    step_clip: float = 0.25,       # gradient step clip (euclidean norm)
) -> Tuple[torch.Tensor, dict]:
    """
    Projected gradient descent with:
      - gradient computed from smooth Huberized path length
      - gradient projected to tangent space of the manifold at each free point
      - explicit projection back to the manifold every 'project_every' iterations
    Endpoints s,t are kept fixed and assumed feasible (g(s)=g(t)=0 or close).
    """
    d = s.numel()
    s = s.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)

    # Initialize path: straight interpolation (then project interior to manifold)
    if init_Y is None:
        alphas = torch.linspace(0., 1., L+1, device=device, dtype=dtype).unsqueeze(-1)  # (L+1,1)
        Y0 = (1 - alphas) * s.unsqueeze(0) + alphas * t.unsqueeze(0)                    # (L+1,d)
        fixed = torch.zeros(L+1, dtype=torch.bool, device=device); fixed[0]=True; fixed[-1]=True
        Y0 = batch_project_to_manifold(Y0, g, fixed_mask=fixed, max_newton=max_newton_projection)
    else:
        assert init_Y.shape == (L+1, d)
        Y0 = init_Y.to(device=device, dtype=dtype)
    Y = nn.Parameter(Y0.clone())

    # Fix endpoints by zeroing grad & position each step
    fixed_mask = torch.zeros(L+1, dtype=torch.bool, device=device); fixed_mask[0]=True; fixed_mask[-1]=True

    opt = torch.optim.SGD([Y], lr=lr, momentum=0.0)

    history = {"loss": []}
    for k in range(steps):
        opt.zero_grad(set_to_none=True)

        # Enforce endpoints before computing loss
        with torch.no_grad():
            Y.data[0]  = s
            Y.data[-1] = t

        loss = path_length_huber(Y, eps=eps_length)
        loss.backward()

        # Project gradient of free points to tangent space at each point
        with torch.no_grad():
            for i in range(1, L):  # interior points only
                Ji = jacobian(g, Y.data[i])                 # (m,d)
                Pi = tangent_project_matrix(Ji)             # (d,d)
                # grad projection
                gi = Y.grad[i] if Y.grad[i] is not None else torch.zeros_like(Y.data[i])
                gi_proj = Pi @ gi
                # optional step clipping for stability
                nrm = torch.linalg.norm(gi_proj)
                if nrm > step_clip:
                    gi_proj = gi_proj * (step_clip / (nrm + 1e-12))
                Y.grad[i].copy_(gi_proj)

            # Zero gradients at endpoints
            Y.grad[0].zero_()
            Y.grad[-1].zero_()

        opt.step()

        # Periodic projection of all free points back to the manifold
        if (k+1) % project_every == 0:
            with torch.no_grad():
                Y.data = batch_project_to_manifold(
                    Y.data, g, fixed_mask=fixed_mask, max_newton=max_newton_projection
                )
                # re-pin endpoints exactly
                Y.data[0]  = s
                Y.data[-1] = t

        history["loss"].append(float(loss.detach().cpu()))
        # Optional: simple LR decay
        if (k+1) % (steps // 3 if steps >= 3 else 1) == 0:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    # Final projection (just in case)
    with torch.no_grad():
        Y.data = batch_project_to_manifold(
            Y.data, g, fixed_mask=fixed_mask, max_newton=max_newton_projection
        )
        Y.data[0]  = s
        Y.data[-1] = t

    return Y.detach(), history


# ---------- Example usage: unit sphere S^{d-1} ----------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # Implicit manifold: unit sphere in R^3 => g(y) = [||y||^2 - 1]
    def g_sphere(y: torch.Tensor) -> torch.Tensor:
        result = y.dot(y) - 1.0
        return result.view(1)  
    d = 3
    # two non-antipodal points on the sphere
    s = torch.tensor([1., 0., 0.])
    t = torch.tensor([0., 1., 0.])

    # Run optimizer
    Y_opt, hist = optimize_path_on_implicit_manifold(
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
        step_clip=0.1
    )

    # Visualize the points Y_opt in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.15, rstride=4, cstride=4, linewidth=0)

    # Plot the path
    Y_np = Y_opt.cpu().numpy()
    ax.plot(Y_np[:,0], Y_np[:,1], Y_np[:,2], marker='o', color='red', label='Geodesic Path')

    # Highlight start and end points
    ax.scatter([Y_np[0,0]], [Y_np[0,1]], [Y_np[0,2]], c='green', s=60, label='Start')
    ax.scatter([Y_np[-1,0]], [Y_np[-1,1]], [Y_np[-1,2]], c='blue', s=60, label='End')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Discrete Geodesic Path on $S^2$')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()

    # Y_opt is an (L+1,3) polyline on S^2 approximating the great-circle arc
    print("Final loss (approx length):", hist["loss"][-1])
    print("First/last points:", Y_opt[0], Y_opt[-1])
