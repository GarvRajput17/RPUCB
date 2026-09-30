"""
Early stopping on a validation metric.

Gameplan v8 §5: monitor val HR@10, patience 10, min_delta 1e-4,
max_epochs 100, restore best.

Replaces the pre-refactor pattern in `train_model`, which ran a fixed
`config['epochs']`, evaluated on the *test* set every epoch, and reported
`max` of each metric independently across all epochs. That had two distinct
problems:

  - model selection read the test set (leak);
  - the reported HR, NDCG, Coverage and ILD could each come from a
    different epoch, so the reported row did not correspond to any single
    checkpoint that ever existed.

This class fixes the first. The second is fixed by reporting all five
metrics from the one restored best checkpoint (see `checkpoint.py`).

Deliberately does not own the model state. It answers "is this the best
epoch so far" and "should we stop"; persisting and restoring weights is
`checkpoint.py`'s job. Keeping them separate means the stopping rule is
testable on a synthetic score sequence with no model in play, which is what
the Checkpoint 4 smoke test does.
"""

from __future__ import annotations


class EarlyStopping:
    """
    Args:
        patience: epochs to allow without a qualifying improvement before
            stopping. 10 per the locked protocol.
        min_delta: how much better the metric must be to count as an
            improvement. 1e-4. Guards against declaring victory on
            floating-point noise, which with patience alone would let a
            plateaued run drift for the full max_epochs.
        mode: 'max' for HR/NDCG-style metrics, 'min' for a loss.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, epoch: int) -> bool:
        """
        Record one epoch's validation score.

        Returns:
            True if this epoch is the new best -- the caller should write a
            checkpoint. False otherwise.

        Sets `self.should_stop` once patience is exhausted; the training
        loop checks that separately so it can log a reason before breaking.
        """
        improved = self._is_improvement(score)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True

        return improved

    def state_dict(self) -> dict:
        """Included in the checkpoint so a resumed run keeps its patience count."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
        self.should_stop = state["should_stop"]

    def summary(self) -> dict:
        """
        Stopping metadata for the results JSON. `stopped_early` distinguishes
        a converged run from one that hit max_epochs -- if a lot of the 135
        runs hit the ceiling, max_epochs=100 was too low and the matrix needs
        rerunning, so this is worth recording rather than reconstructing from
        epoch logs.
        """
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "stopped_early": self.should_stop,
            "epochs_without_improvement": self.epochs_without_improvement,
        }