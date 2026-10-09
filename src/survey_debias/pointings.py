"""Survey layout: ``characterization/{survey}/pointings.list`` and its blocks.

Each survey has its own directory. A block is every ``pointings.list`` line
whose efficiency file is ``{block}.eff``; several such lines are tiles of
one block (the footprint is their union). Multi-epoch surveys keep one
``pointings.list`` and ``{block}.eff`` per ``epoch{i}/`` directory with the
same block tags, and a detection must be found at every epoch::

    characterization/
      {survey}/
        pointings.list   {block}.eff ...           # single epoch
        epoch1/pointings.list  epoch1/{block}.eff  # or one directory per epoch

Line formats follow OSSSSim ``getsur.f95``::

    [rect] width height ra dec jd fill observer eff     # sizes in degrees
    ears ra dec jd fill observer eff                     # MegaCam footprint
    poly n ra dec jd fill observer eff                   # then n lines "dx dy"

RA is decimal degrees or ``hh:mm:ss``; Dec is decimal degrees or
``dd:mm:ss``. ``observer`` is an MPC code or a JPL Horizons vector CSV in
the same directory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .colours import ColourMap, check_filter_name, load_colours
from .efficiency import EpochSelection, SurveySelection, read_efficiency_file
from .grid_bias import FieldView, parse_jpl_horizons_icrf

POINTINGS_NAME = "pointings.list"
# getsur.f95 create_ears half sizes [rad]: body h, w and ear e_h, e_w.
_EARS_H = 0.008678
_EARS_W = 0.008545
_EARS_EH = 0.004169
_EARS_EW = 0.001913
# Older OSSSSim: Detos1 returns the efficiency file name in CHARACTER(10) and
# opens characterization files through CHARACTER(100) paths. Newer OSSSSim
# returns a "survey/block" key (32 characters per part) and allows
# 2048-character paths.
DETOS_SURVEY_NAME_LEN = 10
DETOS_KEY_PART_LEN = 32
OSSSSIM_PATH_MAX = 100
OSSSSIM_KEY_PATH_MAX = 2048


def detos_reports_keys() -> bool:
    """True when the installed ossssim reports ``survey/block`` detection keys."""
    try:
        from ossssim import definitions
    except ImportError:
        return True
    return hasattr(definitions, "SURVEY_KEY_WIDTH")


def ossssim_path_max() -> int:
    return OSSSSIM_KEY_PATH_MAX if detos_reports_keys() else OSSSSIM_PATH_MAX


def _sexagesimal(text: str) -> float:
    """getsur.f95 ``hms``: decimal or ``a:b:c`` with a leading sign."""
    text = text.strip()
    sign = -1.0 if text.startswith("-") else 1.0
    parts = text.lstrip("+-").replace(",", ":").split(":")
    value = 0.0
    for k, part in enumerate(parts[:3]):
        value += float(part) / (60.0 ** k)
    return sign * value


def _ra_deg(text: str) -> float:
    return _sexagesimal(text) * 15.0 if ":" in text else float(text)


@dataclass(frozen=True)
class Tile:
    """One ``pointings.list`` line."""

    ra_deg: float          # bounding-box centre
    dec_deg: float
    width_deg: float       # on-sky extent of the bounding box
    height_deg: float
    area_deg2: float       # footprint area (ossssim when available)
    fill: float
    jd: float
    observer: str
    eff_name: str
    shape: str
    line: int

    @property
    def box_area_deg2(self) -> float:
        return self.width_deg * self.height_deg


def _poly_area(vertices) -> float:
    s = 0.0
    n = len(vertices)
    for k in range(n):
        x1, y1 = vertices[k]
        x2, y2 = vertices[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(0.5 * s)


def parse_pointings(path) -> list[Tile]:
    """Every line of ``pointings.list`` as a :class:`Tile`, in file order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"pointings file not found: {path}")
    lines = path.read_text().splitlines()
    tiles = []
    k = 0
    while k < len(lines):
        raw = lines[k]
        number = k + 1
        k += 1
        if not raw.strip() or raw.startswith("#"):
            continue
        words = raw.split()
        where = f"{path}:{number}"
        try:
            head = words[0].lower()
            if head.startswith("ears"):
                if len(words) < 7:
                    raise ValueError("ears lines need ra dec jd fill observer eff")
                ra, dec = _ra_deg(words[1]), _sexagesimal(words[2])
                rest = words[3:7]
                half_w = math.degrees(_EARS_W + _EARS_EW)
                half_h = math.degrees(_EARS_H)
                width, height = 2.0 * half_w, 2.0 * half_h
                area = 4.0 * _EARS_W * _EARS_H + 4.0 * _EARS_EW * _EARS_EH
                area = area * (180.0 / math.pi) ** 2
                shape = "ears"
            elif head.startswith("poly"):
                if len(words) < 8:
                    raise ValueError("poly lines need n ra dec jd fill observer eff")
                n = int(words[1])
                ra0, dec0 = _ra_deg(words[2]), _sexagesimal(words[3])
                rest = words[4:8]
                vertices = []
                for _ in range(n):
                    if k >= len(lines):
                        raise ValueError(f"poly expects {n} vertex lines")
                    dx, dy = (float(v) for v in lines[k].split()[:2])
                    vertices.append((dx, dy))
                    k += 1
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                dec = dec0 + 0.5 * (min(ys) + max(ys))
                ra = ra0 + 0.5 * (min(xs) + max(xs)) / math.cos(math.radians(dec))
                width, height = max(xs) - min(xs), max(ys) - min(ys)
                area = _poly_area(vertices)
                shape = "poly"
            else:
                if head.startswith("rect"):
                    words = words[1:]
                if len(words) < 8:
                    raise ValueError(
                        "rectangle lines need width height ra dec jd fill observer eff"
                    )
                width, height = float(words[0]), float(words[1])
                ra, dec = _ra_deg(words[2]), _sexagesimal(words[3])
                rest = words[4:8]
                area = width * height
                shape = "rect"
            jd, fill = float(rest[0]), float(rest[1])
        except (ValueError, IndexError) as ex:
            raise ValueError(f"{where}: {ex}") from None
        tiles.append(Tile(
            ra_deg=ra, dec_deg=dec, width_deg=width, height_deg=height,
            area_deg2=area, fill=fill, jd=jd, observer=rest[2],
            eff_name=rest[3], shape=shape, line=number,
        ))
    if not tiles:
        raise ValueError(f"{path}: no pointings")
    return tiles


def check_fortran_paths(directory: Path, tiles: list[Tile],
                        limit: int | None = None) -> None:
    """OSSSSim opens characterization files through fixed-length names."""
    limit = ossssim_path_max() if limit is None else limit
    directory = Path(directory).resolve()
    paths = [directory / POINTINGS_NAME] + [directory / t.eff_name for t in tiles]
    long = [p for p in paths if len(str(p)) > limit]
    if long:
        raise ValueError(
            f"{long[0]} is {len(str(long[0]))} characters; OSSSSim cannot open "
            f"characterization files with paths longer than {limit}. "
            "Move the project (or link it) to a shorter path."
        )


def _ossssim_areas(directory: Path, tiles: list[Tile]) -> list[Tile]:
    """Footprint areas from ossssim's Fortran reader, matched by line order."""
    try:
        from ossssim.survey import SurveyCharacterization
    except ImportError:
        return tiles
    check_fortran_paths(directory, tiles)
    try:
        loaded = SurveyCharacterization.from_directory(directory)
    except Exception as ex:  # noqa: BLE001 - ossssim raises plain RuntimeError/IOError
        raise ValueError(f"{directory}: ossssim could not read pointings.list: {ex}") from ex
    if len(loaded.by_index) != len(tiles):
        raise ValueError(
            f"{directory}: ossssim read {len(loaded.by_index)} pointings, "
            f"pointings.list parser found {len(tiles)}"
        )
    out = []
    for tile, p in zip(tiles, loaded.by_index):
        if Path(p.efnam).name != tile.eff_name:
            raise ValueError(
                f"{directory}: line {tile.line} efficiency {tile.eff_name!r} "
                f"but ossssim read {p.efnam!r}"
            )
        out.append(Tile(**{**tile.__dict__, "area_deg2": float(p.area_deg2)}))
    return out


@dataclass
class Block:
    """All tiles tagged ``{name}.eff`` in one survey, per epoch."""

    survey: str
    name: str
    char_dirs: list            # one directory per epoch
    tiles: list                # tiles[epoch] -> list[Tile]
    selection: SurveySelection = field(repr=False, default=None)

    @property
    def eff_name(self) -> str:
        return f"{self.name}.eff"

    @property
    def detos_name(self) -> str:
        """What older OSSSSim's Detos1 reports: the .eff name, 10 characters."""
        return self.eff_name[:DETOS_SURVEY_NAME_LEN]

    def detos_keys(self) -> list[str]:
        """What newer OSSSSim reports at each epoch: ``{directory}/{block}``."""
        return [f"{Path(d).name}/{self.name}" for d in self.char_dirs]

    @property
    def eff_paths(self) -> list[Path]:
        return [Path(d) / self.eff_name for d in self.char_dirs]

    @property
    def n_epochs(self) -> int:
        return len(self.char_dirs)

    @property
    def filter(self) -> str:
        return self.selection.epochs[0].filter

    @property
    def area_deg2(self) -> float:
        return sum(t.area_deg2 for t in self.tiles[0])

    @property
    def aim_area_deg2(self) -> float:
        return sum(t.box_area_deg2 for t in self.tiles[0])

    def epoch_jds(self) -> tuple:
        return tuple(min(t.jd for t in tiles) for tiles in self.tiles)

    def field_view(self, tile: Tile | None = None, survey=None) -> FieldView:
        """Geometry for the grid_bias helpers: one tile, or the block's box."""
        if tile is None:
            tiles = self.tiles[0]
            ra_lo = min(t.ra_deg - 0.5 * t.width_deg / math.cos(math.radians(t.dec_deg))
                        for t in tiles)
            ra_hi = max(t.ra_deg + 0.5 * t.width_deg / math.cos(math.radians(t.dec_deg))
                        for t in tiles)
            dec_lo = min(t.dec_deg - 0.5 * t.height_deg for t in tiles)
            dec_hi = max(t.dec_deg + 0.5 * t.height_deg for t in tiles)
            dec = 0.5 * (dec_lo + dec_hi)
            ra = 0.5 * (ra_lo + ra_hi)
            width = (ra_hi - ra_lo) * math.cos(math.radians(dec))
            height = dec_hi - dec_lo
            observer = tiles[0].observer
            jds = self.epoch_jds()
        else:
            ra, dec, width, height = tile.ra_deg, tile.dec_deg, tile.width_deg, tile.height_deg
            observer = tile.observer
            jds = (tile.jd,) + self.epoch_jds()[1:]
        rate_cut = self.selection.epochs[0].rate_cut if self.selection else None
        extra = {}
        if survey is not None:
            extra = dict(
                paper_reference_jd=survey.paper_reference_jd,
                check_detected_title=survey.check_detected_title,
            )
        return FieldView(
            name=f"{self.survey}/{self.name}",
            field_ra_deg=ra, field_dec_deg=dec,
            mosaic_width_deg=width, mosaic_height_deg=height,
            epoch_jd=jds, observer_csv=observer,
            rate_cut_min_arcsec_hr=rate_cut[0] if rate_cut else 0.0,
            rate_cut_max_arcsec_hr=rate_cut[1] if rate_cut else 1e9,
            **extra,
        )


def survey_epoch_dirs(char_root, survey: str) -> list[Path]:
    """``characterization/{survey}`` or its ``epoch{i}`` directories."""
    base = Path(char_root) / survey
    if not base.is_dir():
        raise FileNotFoundError(f"survey characterization directory not found: {base}")
    if (base / POINTINGS_NAME).is_file():
        return [base]
    dirs = []
    i = 1
    while (base / f"epoch{i}" / POINTINGS_NAME).is_file():
        dirs.append(base / f"epoch{i}")
        i += 1
    if not dirs:
        raise FileNotFoundError(
            f"{base}: no {POINTINGS_NAME} (or epoch1/{POINTINGS_NAME})"
        )
    return dirs


def load_survey_blocks(char_root, survey: str, use_ossssim: bool = True
                       ) -> dict[str, Block]:
    """Blocks of one survey, keyed by block name (``{block}.eff`` stem)."""
    dirs = survey_epoch_dirs(char_root, survey)
    per_epoch = []
    for d in dirs:
        tiles = parse_pointings(d / POINTINGS_NAME)
        if use_ossssim:
            tiles = _ossssim_areas(d, tiles)
        groups: dict[str, list[Tile]] = {}
        for t in tiles:
            name = Path(t.eff_name).name
            if not name.lower().endswith(".eff"):
                raise ValueError(
                    f"{d / POINTINGS_NAME}:{t.line}: efficiency file {t.eff_name!r} "
                    "must be named {block}.eff"
                )
            groups.setdefault(name[:-4], []).append(t)
        per_epoch.append(groups)
    names = list(per_epoch[0])
    for i, groups in enumerate(per_epoch[1:], start=2):
        if set(groups) != set(names):
            raise ValueError(
                f"characterization/{survey}: epoch{i} blocks "
                f"{sorted(groups)} differ from epoch1 blocks {sorted(names)}"
            )
    blocks = {}
    for name in names:
        block = Block(
            survey=survey, name=name, char_dirs=list(dirs),
            tiles=[groups[name] for groups in per_epoch],
        )
        epochs: list[EpochSelection] = []
        for path in block.eff_paths:
            epochs.append(read_efficiency_file(path))
        filters = {e.filter for e in epochs}
        if len(filters) != 1 or "" in filters:
            raise ValueError(
                f"{survey}/{name}: every {name}.eff needs the same 'filter=' line "
                f"(found {sorted(filters)})"
            )
        block.selection = SurveySelection(epochs)
        blocks[name] = block
    return blocks


def observer_icrf(char_dir, observer: str, jd: float) -> tuple[float, float, float]:
    """Barycentric ICRF observer position [AU] for a pointings.list observer.

    A file in the characterization directory is a JPL Horizons vector CSV;
    anything else is an MPC code, approximated by the Earth's barycentre
    position (the site offset is ~4e-5 AU).
    """
    path = Path(char_dir) / observer
    if path.is_file():
        return parse_jpl_horizons_icrf(path, jd)
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    pos = get_body_barycentric("earth", Time(jd, format="jd", scale="utc"))
    xyz = pos.xyz.to("au").value
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def discover_surveys(char_root) -> list[str]:
    """Survey directories under ``characterization`` with a pointings.list."""
    char_root = Path(char_root)
    if not char_root.is_dir():
        return []
    found = []
    for d in sorted(p for p in char_root.iterdir() if p.is_dir()):
        if (d / POINTINGS_NAME).is_file() or (d / "epoch1" / POINTINGS_NAME).is_file():
            found.append(d.name)
    return found


class Project:
    """Survey characterizations under ``<root>/characterization``, loaded on demand."""

    def __init__(self, root, use_ossssim: bool = True, model_band: str = "r",
                 colour_file: str = "colour.toml"):
        self.root = Path(root)
        self.char_root = self.root / "characterization"
        self.use_ossssim = use_ossssim
        self.model_band = model_band
        self.colour_path = self.root / colour_file
        self._surveys: dict[str, dict[str, Block]] = {}
        self._colours: ColourMap | None = None

    @classmethod
    def for_survey(cls, root, survey, use_ossssim: bool = True) -> "Project":
        return cls(root, use_ossssim=use_ossssim, model_band=survey.model_band,
                   colour_file=survey.colour_file)

    @property
    def colours(self) -> ColourMap:
        if self._colours is None:
            photspec, note = load_colours(self.colour_path)
            self._colours = ColourMap(photspec, self.model_band, note)
        return self._colours

    def colour_for(self, survey: str, block: str, comp=None) -> tuple[str, float, str]:
        """(block filter, filter - model_band, spectral group) for a detection."""
        filt = self.block(survey, block).filter
        colours = self.colours
        return filt, colours.colour(filt, comp), colours.group(comp)

    def survey_names(self, configured=()) -> list[str]:
        """Configured ``surveys``, or every survey directory found."""
        if configured:
            return list(configured)
        return discover_surveys(self.char_root)

    def blocks(self, survey: str) -> dict[str, Block]:
        if survey not in self._surveys:
            blocks = load_survey_blocks(self.char_root, survey, self.use_ossssim)
            for block in blocks.values():
                check_filter_name(block.filter, f"{survey}/{block.eff_name} filter=")
            if detos_reports_keys():
                for block in blocks.values():
                    parts = [block.name] + [Path(d).name for d in block.char_dirs]
                    long = [p for p in parts if len(p) > DETOS_KEY_PART_LEN]
                    if long:
                        raise ValueError(
                            f"characterization/{survey}: {long[0]!r} is longer than "
                            f"{DETOS_KEY_PART_LEN} characters, the most OSSSSim "
                            "allows for a survey directory or block name"
                        )
            else:
                seen: dict[str, str] = {}
                for block in blocks.values():
                    other = seen.setdefault(block.detos_name, block.name)
                    if other != block.name:
                        raise ValueError(
                            f"characterization/{survey}: blocks {other!r} and "
                            f"{block.name!r} share the first {DETOS_SURVEY_NAME_LEN} "
                            "characters of their .eff names, which is all this "
                            "OSSSSim's Detos1 reports; rename one or update ossssim"
                        )
            self._surveys[survey] = blocks
        return self._surveys[survey]

    def block(self, survey: str, block: str) -> Block:
        blocks = self.blocks(survey)
        if block not in blocks:
            raise ValueError(
                f"block {block!r} has no line tagged {block}.eff in "
                f"characterization/{survey}/{POINTINGS_NAME} "
                f"(blocks: {', '.join(sorted(blocks))})"
            )
        return blocks[block]

    def all_blocks(self, surveys) -> list[Block]:
        return [b for s in surveys for b in self.blocks(s).values()]
