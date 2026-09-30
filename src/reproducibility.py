"""
Centralised seeding and determinism control.

Gameplan v8 §6. This module is the *single place* seeding happens. Nothing
else in the codebase should call `manual_seed` / `np.random.seed` /
`random.seed` directly -- in particular `RecDataset.__init__` used to seed
as a side effect of construction, which made the effective seed depend on
object-creation order.

Three gaps this closes, all present in the pre-refactor repo:

  1. `torch.cuda.manual_seed_all` was never called, so any CUDA-side RNG
     (dropout, some kernels) was unseeded.

  2. `num_workers=8` with no `worker_init_fn=` and no `generator=`. PyTorch
     re-seeds `torch` per worker automatically but does NOT touch NumPy or
     the stdlib `random` module. Negative sampling in
     `RecTrainDataset.__getitem__` uses `np.random.randint`, so under the
     default fork start method all 8 workers inherited identical NumPy RNG
     state and emitted the *same* negative samples. That is a correctness
     bug as much as a reproducibility one: it silently collapses negative
     diversity by a factor of `num_workers`.

  3. No cuDNN determinism flags.

Two modes, per the locked protocol:

  seeded-not-bit-exact (default)
      Everything seeded; cuDNN left in benchmark mode. Runs are
      statistically reproducible (same seed -> same trajectory in
      distribution) and fast. This is what the 135-run matrix uses.

  deterministic (`--deterministic`)
      Adds cuDNN deterministic kernels and `use_deterministic_algorithms`.
      Bit-exact at a real throughput cost. Use for debugging and for the
      reproducibility spot-check, not the full matrix.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


# The CUBLAS workspace config is required by `torch.use_deterministic_algorithms`
# for deterministic CUDA matmuls, and must be set before the CUDA context is
# created -- i.e. before any tensor touches the GPU.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed every RNG the training path touches.

    Call exactly once, from `main.py`, before the dataset or model are built.

    Args:
        seed: the run seed. In the matrix this is `base_seed + seed_index`.
        deterministic: if True, additionally force deterministic kernels.
            Bit-exact but slower.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # manual_seed covers the current device only; manual_seed_all covers
        # every visible device. Cheap, and correct under DataParallel.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        _enable_determinism()
    else:
        # Explicitly restore the fast defaults rather than relying on them,
        # so that a deterministic run earlier in the same process cannot
        # leak its settings into a later non-deterministic one.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def is_hip() -> bool:
    """True when torch is a ROCm build, where `torch.cuda.*` is HIP."""
    return getattr(torch.version, "hip", None) is not None


def _enable_determinism() -> None:
    """Strict mode. Separated out so the flag list lives in one place."""
    # CUBLAS_WORKSPACE_CONFIG is a CUDA cuBLAS setting with no meaning under
    # ROCm; setting it there is harmless but misleading in provenance, so it
    # is skipped and recorded as not applicable.
    if not is_hip():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # warn_only=True: a few ops (notably some scatter/index kernels) have no
    # deterministic implementation. Warning rather than raising means a
    # deterministic run degrades loudly instead of failing outright, and the
    # warning lands in the run log where the provenance audit can see it.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader `worker_init_fn`. Seeds NumPy and stdlib `random` inside each
    worker process.

    `torch.initial_seed()` inside a worker returns the per-worker seed that
    PyTorch already derived from the DataLoader's generator, so deriving from
    it keeps NumPy in lockstep with torch *and* gives each worker a distinct
    stream. Masking to 32 bits because `np.random.seed` rejects anything
    wider.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """
    Build the `generator=` passed to DataLoader.

    This is what makes shuffling reproducible and what `seed_worker` derives
    each worker's stream from. Without it, worker seeds come from a global
    RNG whose state depends on however much work happened earlier in the
    process.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def describe_determinism(deterministic: bool) -> dict:
    """
    Determinism metadata for the provenance block in every results JSON
    (§6). Recorded so a run can be audited after the fact rather than
    inferred from the command line.
    """
    return {
        "mode": "deterministic" if deterministic else "seeded",
        "backend": "rocm" if is_hip() else "cuda",
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": (
            "n/a (rocm)" if is_hip() else os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        ),
    }