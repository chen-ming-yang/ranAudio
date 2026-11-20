"""Alignment metrics vs gold char onsets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlignStats:
    n_utt: int = 0
    n_char: int = 0
    n_skip: int = 0
    abs_err: list[float] = field(default_factory=list)
    tol_hit: dict[int, int] = field(default_factory=lambda: {50: 0, 100: 0, 200: 0})

    def add_utt(self, gold_onsets: list[float], pred_onsets: list[float]):
        self.n_utt += 1
        if len(gold_onsets) != len(pred_onsets) or not gold_onsets:
            self.n_skip += 1
            return
        for g, p in zip(gold_onsets, pred_onsets):
            e = abs(float(g) - float(p))
            self.abs_err.append(e)
            self.n_char += 1
            for tol in self.tol_hit:
                if e * 1000 <= tol:
                    self.tol_hit[tol] += 1

    def summary(self) -> dict:
        errs = sorted(self.abs_err)
        n = len(errs)

        def pct(p):
            if n == 0:
                return None
            return errs[min(n - 1, int(p * (n - 1)))]

        return {
            "n_utt": self.n_utt,
            "n_char": self.n_char,
            "n_skip_utt": self.n_skip,
            "onset_mae_s": (sum(errs) / n) if n else None,
            "onset_median_s": pct(0.5),
            "onset_p90_s": pct(0.9),
            "tol_acc": {
                f"{t}ms": (self.tol_hit[t] / n if n else None) for t in sorted(self.tol_hit)
            },
        }
