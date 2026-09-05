"""CRTR BestFSSolver plug-in for a joint latent-motion Sokoban checkpoint.

Deploy this file, sokoban_joint.py, and its model dependencies together on
PYTHONPATH (or in the CRTR repository root). The original solver is unchanged:

    import value_estimator_sokoban_joint
    SolveJob.network = None
    BestFSSolver.value_estimator_class = @ValueEstimatorSokobanJoint
    BestFSSolver.checkpoint_path = '/path/to/joint.pt'

Apply these overrides to CRTR's supplied supervised Sokoban solve config.
``crtr_solve_bindings`` returns the complete override list for runner.py's
``--gin_bindings`` argument. Search uses n_actions=12; greedy/no-search uses 1.
Both retain max_tree_size=6000 and the published config's budget checkpoints.
CRTR reports its @1000-node result with the existing strict ``nodes < 1000``
test in SolveJob.log_results; changing max_tree_size to 1000 is a different run
protocol. This adapter does not modify goal generation or terminal-state logic.

Inputs are flattened categorical boards (144 tiles, values 0..6), including
CRTR's integer-valued float32 arrays. Encoding and the sole integrated-motion
distance readout belong to the checkpoint's model. The generic ``model``,
``metric`` and ``include_actions`` constructor arguments are accepted for the
solver interface; the checkpoint supplies the actual network and readout.
"""

import gin
import torch

if __package__:
    from .sokoban_joint import load_joint_checkpoint
else:
    from sokoban_joint import load_joint_checkpoint


def crtr_solve_bindings(checkpoint_path, mode="search", n_jobs=1000):
    """Overrides for the existing supervised config; no solver fork is needed."""
    if mode not in ("search", "greedy"):
        raise ValueError("mode must be 'search' or 'greedy'")
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return [
        "import value_estimator_sokoban_joint",
        "SolveJob.network = None",
        "BestFSSolver.value_estimator_class = @ValueEstimatorSokobanJoint",
        f"BestFSSolver.checkpoint_path = {str(checkpoint_path)!r}",
        f"SolveJob.n_actions = {12 if mode == 'search' else 1}",
        "BestFSSolver.max_tree_size = 6000",
        "BestFSSolver.max_tree_depth = -1",
        f"SolveJob.n_jobs = {n_jobs}",
        "SolveJob.n_parallel_workers = 1",
        "SolveJob.batch_size = 200",
        "SolveJob.budget_checkpoints = [50, 100, 500, 1000]",
    ]


@gin.configurable
class ValueEstimatorSokobanJoint:
    """Use the trained joint model as CRTR's direct state-to-goal heuristic."""

    def __init__(self, model=None, metric=None, include_actions=False,
                 checkpoint_path=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.checkpoint = None

    def construct_networks(self):
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for ValueEstimatorSokobanJoint")
        self.model, self.checkpoint = load_joint_checkpoint(self.checkpoint_path,
                                                           device=self.device)
        self.model.to(self.device).eval()
        config = self.checkpoint.get("config", {})
        print(f"ValueEstimatorSokobanJoint loaded {self.checkpoint_path} "
              f"(width={config.get('width')} T={config.get('T')})", flush=True)

    def _flat(self, board):
        return torch.as_tensor(board, device=self.device).reshape(1, -1).to(torch.uint8)

    def _require_model(self):
        if self.model is None:
            raise RuntimeError("Call construct_networks before requesting distances")

    def _predict(self, states, goals):
        self._require_model()
        with torch.inference_mode():
            distances = self.model(states, goals).reshape(-1)
        if distances.numel() != states.shape[0]:
            raise RuntimeError("Checkpoint model must return one distance per input state")
        # reshape(-1), never squeeze(): a batch of one must remain indexable.
        return distances.cpu()

    def get_solved_distance(self, state_str, goal, action_in=None):
        return self._predict(self._flat(state_str), self._flat(goal))

    def get_solved_distance_batch(self, states, goal):
        self._require_model()
        batch = len(states)
        if batch == 0:
            return torch.empty(0, dtype=torch.float32)
        inputs = torch.as_tensor(states, device=self.device).reshape(batch, -1).to(torch.uint8)
        goals = self._flat(goal).expand(batch, -1)
        return self._predict(inputs, goals)


__all__ = ["ValueEstimatorSokobanJoint", "crtr_solve_bindings"]
