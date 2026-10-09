"""Per-survey bias tables ``characterization/{survey}/bias_grid[_rih].csv``.

One row per (block, colour group, cell): the Monte Carlo estimate of
P(detect in that block | cell), the number of aimed draws and detections
behind it, and its standard error.
"""
from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

CELL_FIELDS = {
    "model_ae": ("r_bin", "i_bin", "h_bin"),
    "aq_grid": ("a_bin", "q_bin", "si_bin", "h_bin"),
}
BIAS_FILE_FIELDS = ("block", "colour_group")


def bias_file_name(method: str) -> str:
    return "bias_grid_rih.csv" if method == "model_ae" else "bias_grid.csv"


@dataclass
class BiasEstimate:
    """Monte Carlo P(detect in block | cell) and its standard error."""

    bias: float
    n_drawn: int
    n_detected: int | None = None
    bias_se: float | None = None
    sampled: dict = field(default=None, repr=False)
    detected: dict = field(default=None, repr=False)

    def relative_error(self, fallback: float | None = None) -> float | None:
        """se/bias; else 1/sqrt(n_detected); else ``fallback``."""
        if self.bias > 0.0 and self.bias_se is not None:
            return self.bias_se / self.bias
        if self.n_detected:
            return 1.0 / math.sqrt(self.n_detected)
        return fallback


def mc_bias(n_aimed: int, sum_w: float, sum_w2: float) -> tuple[float, float]:
    """Mean of x = 1_detected * P_geom over aimed draws and its MC error."""
    if n_aimed <= 0:
        return 0.0, 0.0
    bias = sum_w / n_aimed
    var = max(sum_w2 / n_aimed - bias * bias, 0.0)
    return float(bias), float(math.sqrt(var / n_aimed))


def _float_or_none(text):
    text = "" if text is None else str(text).strip()
    return float(text) if text else None


def parse_cell(text) -> tuple:
    """Cell key from its results-file text, e.g. ``(38.0, 10.0, 10.9)``."""
    value = ast.literal_eval(str(text).strip())
    return tuple(round(float(v), 6) for v in value)


def load_bias_file(path, method: str) -> dict:
    """``(block, colour_group, cell) -> BiasEstimate`` from a survey's bias file."""
    path = Path(path)
    if not path.exists():
        return {}
    cell_fields = CELL_FIELDS[method]
    out = {}
    with path.open() as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        fields = reader.fieldnames or []
        missing = [f for f in (*BIAS_FILE_FIELDS, *cell_fields, "bias", "n_drawn")
                   if f not in fields]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        for row in reader:
            cell = tuple(round(float(row[k]), 6) for k in cell_fields)
            n_det = _float_or_none(row.get("n_detected"))
            out[(row["block"], row["colour_group"], cell)] = BiasEstimate(
                bias=float(row["bias"]),
                n_drawn=int(float(row["n_drawn"])),
                n_detected=None if n_det is None else int(n_det),
                bias_se=_float_or_none(row.get("bias_se")),
            )
    return out


def save_bias_file(path, entries: dict, method: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*BIAS_FILE_FIELDS, *CELL_FIELDS[method],
                    "bias", "n_drawn", "n_detected", "bias_se"])
        for (block, group, cell), est in sorted(entries.items()):
            w.writerow([
                block, group, *cell, f"{est.bias:.9g}", est.n_drawn,
                "" if est.n_detected is None else est.n_detected,
                "" if est.bias_se is None else f"{est.bias_se:.6g}",
            ])


def load_legacy_cache(path, method: str) -> dict:
    """``cell -> BiasEstimate`` from a pre-survey root ``bias_grid[_rih].csv``."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "block" in fields:
            return {}
        keys = CELL_FIELDS[method]
        if any(k not in fields for k in keys):
            return {}
        for row in reader:
            cell = tuple(round(float(row[k]), 6) for k in keys)
            out[cell] = BiasEstimate(float(row["bias"]), int(float(row["n_drawn"])))
    return out


def load_project_bias(char_root, surveys, method: str) -> dict:
    """``(survey, block, colour_group, cell) -> BiasEstimate`` for every survey."""
    out = {}
    for s in surveys:
        path = Path(char_root) / s / bias_file_name(method)
        for (blk, g, c), est in load_bias_file(path, method).items():
            out[(s, blk, g, c)] = est
    return out
