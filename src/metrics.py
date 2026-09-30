"""
Full-catalog ranking metrics.

Gameplan v8 §5: HR@10, HR@100, NDCG@10, Coverage@10, ILD@10 -- all five
derived from one ranked list per user, over the full catalog minus that
user's train items (and minus the val item at test time).

Why full-catalog (§9): Krichene & Rendle, KDD'20 -- metrics computed
against a sampled negative set do not preserve model ordering even in
expectation. Two of the five were not measurable under the old protocol at
all. HR@100 was identically 1.0 when there were only 100 candidates, and
Coverage@10 described sampled candidate pools rather than the catalog.

TIE HANDLING IS NOT A DETAIL HERE.

The first-light run made this concrete. An item with no training
interactions has an all-zero interaction column, so every model's item
tower maps it to one shared embedding and every such item takes the same
score for a given user. A user with no training interactions is worse: for
the MIND family the interest vectors come out exactly zero, so every item
in the catalog scores exactly 0.0.

Under the optimistic convention -- rank = 1 + |{strictly greater}| -- both
cases are scored as rank 1. On AMusic that handed MIND 43 guaranteed hits
from its 43 train-less users, 65% of its measured HR@10, before training
contributed anything. And roughly 26% of AMusic test positives and 30% of
AToy's are cold items that tie with each other in every model.

The damage is worse than a constant offset, because cold fractions rise
with sparsity: 0% on lastfm, 0.03% on ml-1m, 0.1% on citeulike-a, 26% on
AMusic, 30% on AToy. "RP-UCB helps more as sparsity increases" is the
central hypothesis, so an artifact that grows along that exact axis would
be indistinguishable from the effect being tested.

The default is therefore `mid`: rank = 1 + |greater| + |tied| / 2, the
expected rank under random tie-breaking. It is unbiased, and it makes both
cold cases misses for every model equally rather than free hits for
whichever model's constant happens to sit high. `optimistic` and
`pessimistic` remain available for sensitivity analysis -- the gap between
optimistic and mid on a given run *is* the size of the artifact.

Tie detection uses a tolerance rather than exact equality. Identical
inputs can differ in the last ulp when they land in different chunks of a
batched matmul, so cold items that ought to tie exactly sometimes differ
by ~1e-7; exact `==` would miss those and silently fall back to optimistic
behaviour for them.

Because cold positives are unrankable by construction rather than by model
quality, every metric is also reported over the warm subset, so the
headline number and the number that isolates what models can actually
rank are both on the record.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_HR_KS = (10, 100)
DEFAULT_NDCG_KS = (10,)
DEFAULT_COVERAGE_K = 10
DEFAULT_ILD_K = 10

TIE_POLICIES = ("mid", "optimistic", "pessimistic")
DEFAULT_TIE_POLICY = "mid"
DEFAULT_TIE_ATOL = 1e-6


def ranks_of_positives(
    scores: torch.Tensor,
    positives: torch.Tensor,
    tie_policy: str = DEFAULT_TIE_POLICY,
    tie_atol: float = DEFAULT_TIE_ATOL,
) -> torch.Tensor:
    """
    Rank of each user's held-out item within the full-catalog scores.

    Args:
        scores: [B, num_items]. Excluded items (train history, and the val
            item at test time) must already be -inf: this function does not
            know what to exclude, and ranking against a user's own training
            items is the easiest way to manufacture good numbers.
        positives: [B] held-out item id per user.
        tie_policy: see the module docstring.

    Returns:
        [B] float ranks, 1-indexed. Float because `mid` produces half-integers.
    """
    if tie_policy not in TIE_POLICIES:
        raise ValueError(f"tie_policy must be one of {TIE_POLICIES}, got {tie_policy!r}")

    pos_scores = scores.gather(1, positives.unsqueeze(1))          # [B, 1]
    diff = scores - pos_scores

    n_greater = (diff > tie_atol).sum(dim=1).float()
    # -1 drops the positive, which always ties with itself.
    n_tied = (diff.abs() <= tie_atol).sum(dim=1).float() - 1.0

    if tie_policy == "optimistic":
        return 1.0 + n_greater
    if tie_policy == "pessimistic":
        return 1.0 + n_greater + n_tied
    return 1.0 + n_greater + n_tied / 2.0


def count_ties(
    scores: torch.Tensor,
    positives: torch.Tensor,
    tie_atol: float = DEFAULT_TIE_ATOL,
) -> torch.Tensor:
    """
    [B] number of non-excluded items sharing the positive's score.

    Reported per run: a large mean here says the ranking is being decided
    by the tie convention rather than by the model, which is the signature
    of the cold-item problem.
    """
    pos_scores = scores.gather(1, positives.unsqueeze(1))
    return ((scores - pos_scores).abs() <= tie_atol).sum(dim=1).float() - 1.0


def hit_rate_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    return (ranks <= k).float()


def ndcg_at_k(ranks: torch.Tensor, k: int) -> torch.Tensor:
    """
    One held-out relevant item per user (leave-one-out), so IDCG is 1 and
    NDCG is 1/log2(rank + 1) inside the cutoff, 0 outside.
    """
    gains = 1.0 / torch.log2(ranks.float() + 1.0)
    return torch.where(ranks <= k, gains, torch.zeros_like(gains))


def intra_list_diversity(topk_items: torch.Tensor, item_vectors: torch.Tensor) -> torch.Tensor:
    """
    Mean pairwise cosine distance within each user's top-k list.

    `item_vectors` is the co-interaction column matrix: metadata-free and
    identically defined across all five datasets, which is what makes ILD
    comparable between them as the offline stand-in for Pinterest's
    A-Pincepts measure (§9). Note that on sparse data most item pairs share
    no users at all, so ILD sits near 1 for every model and discriminates
    weakly -- it partly measures popularity, since popular items co-occur
    more and therefore score lower.
    """
    B, k = topk_items.shape
    if k < 2:
        return torch.zeros(B, device=topk_items.device)

    vecs = F.normalize(item_vectors[topk_items].float(), dim=-1)      # [B, k, D]
    sim = torch.bmm(vecs, vecs.transpose(1, 2))                        # [B, k, k]

    eye = torch.eye(k, device=sim.device).unsqueeze(0)
    mean_sim = (sim * (1.0 - eye)).sum(dim=(1, 2)) / (k * (k - 1))
    return 1.0 - mean_sim


class FullCatalogMetrics:
    """
    Streaming accumulator over user batches.

    Full-catalog scoring produces [B, num_items] per batch; each batch is
    reduced to per-user scalars plus a set of surfaced item ids, so nothing
    of size [num_users, num_items] is ever held.

    `warm_mask` on each update marks the users whose result reflects model
    quality: the positive has at least one training interaction, and the
    user has some training history. Everything is accumulated twice, once
    over all users and once over that subset.

    Usage:
        acc = FullCatalogMetrics(num_items, item_vectors=cols)
        for batch in loader:
            acc.update(scores, positives, warm_mask=warm)
        results = acc.compute()
    """

    def __init__(
        self,
        num_items: int,
        item_vectors: torch.Tensor | None = None,
        hr_ks: tuple[int, ...] = DEFAULT_HR_KS,
        ndcg_ks: tuple[int, ...] = DEFAULT_NDCG_KS,
        coverage_k: int = DEFAULT_COVERAGE_K,
        ild_k: int = DEFAULT_ILD_K,
        tie_policy: str = DEFAULT_TIE_POLICY,
        tie_atol: float = DEFAULT_TIE_ATOL,
    ) -> None:
        self.num_items = num_items
        self.item_vectors = item_vectors
        self.hr_ks = tuple(hr_ks)
        self.ndcg_ks = tuple(ndcg_ks)
        self.coverage_k = coverage_k
        self.ild_k = ild_k
        self.tie_policy = tie_policy
        self.tie_atol = tie_atol

        self._sums = {"all": self._zero_sums(), "warm": self._zero_sums()}
        self._n = {"all": 0, "warm": 0}
        self._surfaced: set[int] = set()
        self._tie_total = 0.0
        self._positives_with_ties = 0

    def _zero_sums(self):
        return {
            **{f"HR@{k}": 0.0 for k in self.hr_ks},
            **{f"NDCG@{k}": 0.0 for k in self.ndcg_ks},
            "ILD": 0.0,
        }

    @torch.no_grad()
    def update(self, scores, positives, warm_mask: torch.Tensor | None = None) -> None:
        """
        Args:
            scores: [B, num_items], exclusions already applied as -inf.
            positives: [B] held-out item ids.
            warm_mask: [B] bool, True where the positive is rankable in
                principle. None treats every user as warm.
        """
        ranks = ranks_of_positives(scores, positives, self.tie_policy, self.tie_atol)
        ties = count_ties(scores, positives, self.tie_atol)
        B = ranks.size(0)

        top_k = min(max(self.coverage_k, self.ild_k), scores.size(1))
        _, topk_idx = torch.topk(scores, top_k, dim=1)

        ild = None
        if self.item_vectors is not None and self.ild_k >= 2:
            ild = intra_list_diversity(topk_idx[:, : self.ild_k], self.item_vectors)

        for group, mask in (("all", None), ("warm", warm_mask)):
            if mask is None and group == "warm":
                mask = torch.ones(B, dtype=torch.bool, device=ranks.device)
            sel = ranks if mask is None else ranks[mask]
            if sel.numel() == 0:
                continue
            for k in self.hr_ks:
                self._sums[group][f"HR@{k}"] += hit_rate_at_k(sel, k).sum().item()
            for k in self.ndcg_ks:
                self._sums[group][f"NDCG@{k}"] += ndcg_at_k(sel, k).sum().item()
            if ild is not None:
                self._sums[group]["ILD"] += (ild if mask is None else ild[mask]).sum().item()
            self._n[group] += sel.numel()

        # Coverage is over all evaluated users, matching the headline metrics.
        self._surfaced.update(topk_idx[:, : self.coverage_k].reshape(-1).tolist())
        self._tie_total += ties.sum().item()
        self._positives_with_ties += (ties > 0).sum().item()

    def compute(self) -> dict:
        """
        The five reported metrics over all users, the same five over the
        warm subset with a `_warm` suffix, and tie diagnostics.

        Coverage is a property of the accumulated run rather than a mean of
        per-user values, which is why it cannot be averaged like the rest.
        """
        if self._n["all"] == 0:
            raise RuntimeError("compute() called before any update()")

        n_all = float(self._n["all"])
        out: dict = {}

        for k in self.hr_ks:
            out[f"HR@{k}"] = self._sums["all"][f"HR@{k}"] / n_all
        for k in self.ndcg_ks:
            out[f"NDCG@{k}"] = self._sums["all"][f"NDCG@{k}"] / n_all

        out[f"Coverage@{self.coverage_k}"] = (
            len(self._surfaced) / self.num_items if self.num_items else 0.0
        )
        out[f"ILD@{self.ild_k}"] = (
            self._sums["all"]["ILD"] / n_all if self.item_vectors is not None else float("nan")
        )
        out["num_users_evaluated"] = self._n["all"]

        n_warm = float(self._n["warm"])
        out["num_users_warm"] = self._n["warm"]
        if n_warm:
            for k in self.hr_ks:
                out[f"HR@{k}_warm"] = self._sums["warm"][f"HR@{k}"] / n_warm
            for k in self.ndcg_ks:
                out[f"NDCG@{k}_warm"] = self._sums["warm"][f"NDCG@{k}"] / n_warm

        out["tie_policy"] = self.tie_policy
        out["mean_ties_at_positive"] = self._tie_total / n_all
        out["frac_positives_with_ties"] = self._positives_with_ties / n_all

        return out