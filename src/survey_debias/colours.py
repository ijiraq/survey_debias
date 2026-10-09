"""Object colours from ``colour.toml``, in the ``ossssim.color.PhotSpec`` layout.

``colour.toml`` has one table per spectral group, each mapping band ratios
to colours in magnitudes, for example::

    [default]
    "g-g" = 0.0
    "r-g" = -0.7

    [cold]
    "g-g" = 0.0
    "r-g" = -0.85

The colour that maps a model-band H onto a block's filter is
``filter - model_band`` for the object's spectral group, the same term
OSSSSim's ``detos1`` applies. The spectral group is chosen from the
detection's ``comp`` value with ``PhotSpec.orbital_to_spectral_group``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from astropy import units

try:
    from ossssim.color import PhotSpec
except ImportError:  # pragma: no cover - ossssim is a hard dependency
    PhotSpec = None


def check_filter_name(name: str, where: str) -> str:
    """OSSSSim indexes filters by one ASCII character between A and z."""
    text = str(name).strip()
    if len(text) != 1 or not ("A" <= text <= "z"):
        raise ValueError(
            f"{where}: filter {name!r} must be one character A-z "
            "(OSSSSim filter_to_index); give the band a letter in the .eff "
            "file and the matching colours in colour.toml"
        )
    return text


def load_colours(path) -> tuple["PhotSpec", str]:
    """``PhotSpec`` from ``colour.toml``; ossssim's built-in colours if absent."""
    path = Path(path)
    if not path.is_file():
        return PhotSpec(), f"colour file {path} not found; using ossssim PhotSpec.COLORS"
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not data:
        raise ValueError(f"{path}: no spectral groups")
    colours = {}
    for group, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(
                f"{path}: {group!r} must be a table of band ratios, "
                'for example [default] "r-g" = -0.7'
            )
        ratios = {}
        for ratio, value in table.items():
            this, sep, base = str(ratio).partition("-")
            if not sep:
                raise ValueError(f"{path}: [{group}] {ratio!r} is not a band ratio like \"r-g\"")
            check_filter_name(this, f"{path} [{group}] {ratio}")
            check_filter_name(base, f"{path} [{group}] {ratio}")
            try:
                ratios[f"{this}-{base}"] = float(value) * units.mag
            except (TypeError, ValueError):
                raise ValueError(f"{path}: [{group}] {ratio} = {value!r} is not a number") from None
        colours[str(group)] = ratios
    if "default" not in colours:
        raise ValueError(f"{path}: needs a [default] spectral group")
    return PhotSpec(colours), f"colours from {path}"


class ColourMap:
    """``filter - model_band`` colours for each spectral group."""

    def __init__(self, photspec: "PhotSpec", model_band: str, note: str = ""):
        self.photspec = photspec
        self.model_band = check_filter_name(model_band, "model_band")
        self.note = note
        self._cache: dict[str, dict[str, float]] = {}

    def group(self, comp) -> str:
        return self.photspec.orbital_to_spectral_group(str(comp or ""))

    def _rebased(self, group: str) -> dict[str, float]:
        if group not in self._cache:
            table = self.photspec.colors[group]
            bases = {ratio.split("-")[1] for ratio in table}
            for base in bases:
                if f"{self.model_band}-{base}" not in table:
                    raise ValueError(
                        f"colour group [{group}] has no {self.model_band}-{base} "
                        f"colour, so model_band {self.model_band!r} cannot be reached"
                    )
            specphot = self.photspec.transform_spectral_group_to_model_band(
                group, self.model_band)
            self._cache[group] = {
                ratio.split("-")[0]: float(getattr(v, "value", v))
                for ratio, v in specphot.items()
            }
        return self._cache[group]

    def colour(self, filt: str, comp=None) -> float:
        """``filter - model_band`` for the spectral group matching ``comp``."""
        group = self.group(comp)
        table = self._rebased(group)
        if filt not in table:
            raise ValueError(
                f"colour group [{group}] has no colour for filter {filt!r} "
                f"(have {', '.join(sorted(table))})"
            )
        return table[filt]

    def validate(self, filters) -> None:
        """Every filter must have a colour in every spectral group."""
        for group in self.photspec.spectral_groups:
            table = self._rebased(group)
            missing = sorted(set(filters) - set(table))
            if missing:
                raise ValueError(
                    f"colour group [{group}] has no colour for filter(s) "
                    f"{', '.join(missing)}"
                )

    def colors_list(self, comp) -> list:
        """The Fortran colour array ossssim passes to ``detos1``."""
        return self.photspec.colors_list(str(comp or ""), self.model_band)
