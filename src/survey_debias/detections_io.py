"""OSSOS / CFEPS ``.detections`` files: read, assign blocks, select a sample.

The release format (``characterization/OSSOS_v11/README.formats``) is
whitespace separated; the column names are on the last ``#`` line:

    cl p j k sh object mag e_mag Filt Hsur dist e_dist Nobs time av_xres
    av_yres max_x max_y a e_a e e_e i e_i Omega e_Omega omega e_omega tperi
    e_tperi RAdeg DEdeg JD rate MPC

``cl`` is the dynamical class (cla, res, sca, det, cen, jco, xxx) and, for
classicals, ``p`` the part of the belt (i inner, m main, o outer). There is
no block column: OSSOS and MA encode the block in the designation, CFEPS and
HiLat use the block name as a prefix, and the rest are placed by RA, Dec and
JD.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .pointings import Block, load_survey_blocks

NUMERIC = {
    "j", "k", "mag", "e_mag", "Hsur", "dist", "e_dist", "Nobs", "time", "av_xres",
    "av_yres", "max_x", "max_y", "a", "e_a", "e", "e_e", "i", "e_i", "Omega",
    "e_Omega", "omega", "e_omega", "tperi", "e_tperi", "RAdeg", "DEdeg", "JD", "rate",
}

# Designation prefix -> block, where the block is not a prefix of the name.
BLOCK_PREFIXES = {
    "OSSOS_v11": {
        "o3e": "2013AE", "o3o": "2013AO", "o3l": "2013BL", "o4h": "2014BH",
        "o5p": "2015AP", "o5m": "2015AM", "o5s": "2015BS", "o5t": "2015BT",
        "o5c": "2015BC", "o5d": "2015BD",
    },
    "MA": {"mal": "ma11la", "mah": "ma12ha"},
}

CSV_COLUMNS = ["name", "survey", "block", "filter", "mag", "e_mag", "d_bary", "a", "e",
               "i", "ifree", "ifree_source", "comp", "cl", "p", "sh", "Hsur", "ra", "dec",
               "jd", "rate"]


def read_detections_file(path) -> list[dict]:
    """Rows of a ``.detections`` file, keyed by the last ``#`` header line."""
    path = Path(path)
    header = None
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            words = line.lstrip("#").split()
            if words:
                header = words
            continue
        if header is None:
            raise ValueError(f"{path}: data before the column header line")
        values = line.split()
        if len(values) < len(header):
            raise ValueError(f"{path}: {len(values)} values for {len(header)} columns: {line}")
        row = {}
        for key, value in zip(header, values):
            row[key] = float(value) if key in NUMERIC else value
        rows.append(row)
    if header is None:
        raise ValueError(f"{path}: no column header")
    return rows


def _angular_sep_deg(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def assign_block(name: str, blocks: dict[str, Block], prefixes: dict | None = None,
                 ra: float | None = None, dec: float | None = None,
                 jd: float | None = None, max_days: float = 60.0) -> str:
    """Block of a detection, from its designation or else its discovery pointing.

    In order: an explicit designation prefix; the longest block name (without
    ``-smooth``) that prefixes the designation; the only block; the block with
    the nearest tile within ``max_days`` of the discovery JD that contains the
    discovery position.
    """
    low = name.lower()
    for prefix, block in sorted((prefixes or {}).items(), key=lambda kv: -len(kv[0])):
        if low.startswith(prefix.lower()) and block in blocks:
            return block
    stems = [(b.lower().removesuffix("-smooth"), b) for b in blocks]
    matches = sorted((s, b) for s, b in stems if low.startswith(s))
    if matches:
        longest = max(len(s) for s, _ in matches)
        best = [b for s, b in matches if len(s) == longest]
        if len(best) == 1:
            return best[0]
    if len(blocks) == 1:
        return next(iter(blocks))
    if None in (ra, dec, jd):
        raise ValueError(f"{name}: no block from its designation and no RA/Dec/JD")
    candidates = []
    for block in blocks.values():
        for tile in block.tiles[0]:
            if abs(tile.jd - jd) > max_days:
                continue
            sep = _angular_sep_deg(ra, dec, tile.ra_deg, tile.dec_deg)
            reach = 0.5 * math.hypot(tile.width_deg, tile.height_deg)
            if sep <= reach:
                candidates.append((abs(tile.jd - jd), sep, block.name))
    if not candidates:
        raise ValueError(f"{name}: no tile within {max_days} d contains "
                         f"RA={ra:.4f} Dec={dec:.4f}")
    return min(candidates)[2]


def _unpack_mpc(packed: str) -> str:
    """Release MPC column (packed provisional or packed number) -> compact designation."""
    from .ifree import NCODE, unpack_provisional

    packed = (packed or "").strip()
    if packed.isdigit():
        return str(int(packed))
    if len(packed) == 5 and packed[0] in NCODE[10:] and packed[1:].isdigit():
        return str(NCODE.index(packed[0]) * 10000 + int(packed[1:]))
    return unpack_provisional(packed)


def release_ifree(row: dict) -> tuple[float, str]:
    """(I_free, source) of a release row: Huang et al. Table 1, else their forcing-pole grid.

    Table 1 is looked up by the release name (OSSOS++ designations) and the MPC
    designation; otherwise the grid is applied to the release's barycentric
    elements (a, e, i, Omega, omega).
    """
    from .ifree import free_inclination, huang_lookup

    hit = huang_lookup(row.get("object"), _unpack_mpc(row.get("MPC", "")))
    if hit is not None and math.isfinite(hit["Ifree"]):
        return float(hit["Ifree"]), "huang2022_table"
    try:
        ifree, _ = free_inclination(float(row["a"]), float(row["e"]), float(row["i"]),
                                    float(row["Omega"]), float(row["omega"]))
    except (KeyError, ValueError):
        return math.nan, "none"
    return float(ifree), "huang2022_grid"


@dataclass
class ColdCut:
    """Sample selection on the release columns (the same cut is applied to the model).

    ``i_max`` cuts the ecliptic inclination, ``ifree_max`` the free
    inclination (``release_ifree``); either may be None.
    """

    classes: tuple = ("cla",)
    parts: tuple = ("m",)
    i_max: float | None = 5.0
    secure_only: bool = False
    ifree_max: float | None = None

    def __call__(self, row: dict) -> bool:
        if row["cl"] not in self.classes:
            return False
        if self.parts and row["p"] not in self.parts:
            return False
        if self.secure_only and not str(row["sh"]).startswith("S"):
            return False
        if self.i_max is not None and not float(row["i"]) < self.i_max:
            return False
        if self.ifree_max is not None:
            if "ifree" not in row:
                row["ifree"], row["ifree_source"] = release_ifree(row)
            if not float(row["ifree"]) < self.ifree_max:
                return False
        return True

    def describe(self) -> str:
        parts = [f"cl in {self.classes}", f"p in {self.parts or 'any'}"]
        if self.i_max is not None:
            parts.append(f"i < {self.i_max:g} deg")
        if self.ifree_max is not None:
            parts.append(f"i_free < {self.ifree_max:g} deg")
        if self.secure_only:
            parts.append("secure classes only")
        return ", ".join(parts)


def convert(char_root, survey: str, cut: ColdCut, path=None) -> list[dict]:
    """CSV rows for one survey: every detection, ``comp`` = cold where it passes ``cut``."""
    char_root = Path(char_root)
    path = Path(path) if path else char_root / survey / f"{survey}.detections"
    blocks = load_survey_blocks(char_root, survey, use_ossssim=False)
    prefixes = BLOCK_PREFIXES.get(survey)
    out = []
    for row in read_detections_file(path):
        name = row["object"]
        block = assign_block(name, blocks, prefixes, row["RAdeg"], row["DEdeg"], row["JD"])
        filt = blocks[block].filter
        cold = cut(row)
        if "ifree" not in row:
            row["ifree"], row["ifree_source"] = release_ifree(row)
        out.append({
            "name": name, "survey": survey, "block": block, "filter": filt,
            "mag": row["mag"], "e_mag": row["e_mag"], "d_bary": row["dist"],
            "a": row["a"], "e": row["e"], "i": row["i"],
            "ifree": row["ifree"], "ifree_source": row["ifree_source"],
            "comp": "cold" if cold else "other",
            "cl": row["cl"], "p": row["p"], "sh": row["sh"], "Hsur": row["Hsur"],
            "ra": row["RAdeg"], "dec": row["DEdeg"], "jd": row["JD"], "rate": row["rate"],
        })
    return out


def read_duplicates(path) -> list[tuple[str, ...]]:
    """Groups of designations of one object (one group per line, ``#`` comments)."""
    groups = []
    for raw in Path(path).read_text().splitlines():
        words = raw.split("#", 1)[0].split()
        if len(words) >= 2:
            groups.append(tuple(words))
    return groups


def drop_duplicates(names: dict, groups, priority) -> dict:
    """Designations to drop per survey, so each object is counted once.

    ``names`` maps survey -> designations in its sample; an object found by
    several surveys is kept in the first of ``priority`` that has it.
    """
    order = {s: k for k, s in enumerate(priority)}
    where = {}
    for survey, members in names.items():
        for n in members:
            where.setdefault(n, []).append(survey)
    drop = {s: set() for s in names}
    for group in groups:
        found = [(order.get(s, len(order)), s, n) for n in group for s in where.get(n, [])]
        if len(found) < 2:
            continue
        found.sort()
        for _, s, n in found[1:]:
            drop[s].add(n)
    return drop


def write_csv(rows: list[dict], path, note: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if note:
            for line in note.splitlines():
                fh.write(f"# {line}\n")
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="survey_debias_detections",
        description="Convert an OSSOS/CFEPS .detections file to the CSV survey_debias "
                    "reads, with a survey and block per detection and comp = cold "
                    "for the selected sample.",
    )
    p.add_argument("--root", default=".", help="directory holding characterization/")
    p.add_argument("--survey", required=True, help="characterization/{survey}")
    p.add_argument("--detections", default=None,
                   help="input file (default characterization/{survey}/{survey}.detections)")
    p.add_argument("--out", default=None,
                   help="output CSV (default characterization/{survey}/{survey}.csv)")
    p.add_argument("--classes", default="cla", help="comma-separated cl values (default cla)")
    p.add_argument("--parts", default="m", help="comma-separated p values ('' for any)")
    p.add_argument("--i-max", type=float, default=None,
                   help="ecliptic inclination cut [deg] (default 5 unless --ifree-max)")
    p.add_argument("--ifree-max", type=float, default=None,
                   help="free inclination cut [deg] (Huang et al. 2022), e.g. 4")
    p.add_argument("--secure-only", action="store_true", help="only sh = S... classes")
    args = p.parse_args(argv)
    i_max = args.i_max if args.i_max is not None or args.ifree_max is not None else 5.0
    cut = ColdCut(tuple(c for c in args.classes.split(",") if c),
                  tuple(c for c in args.parts.split(",") if c), i_max, args.secure_only,
                  args.ifree_max)
    char_root = Path(args.root) / "characterization"
    rows = convert(char_root, args.survey, cut, args.detections)
    out = Path(args.out) if args.out else char_root / args.survey / f"{args.survey}.csv"
    n_cold = sum(r["comp"] == "cold" for r in rows)
    note = f"{args.survey} detections; comp = cold for {cut.describe()}"
    write_csv(rows, out, note)
    blocks = sorted({r["block"] for r in rows})
    print(f"wrote {out}: {len(rows)} detections, {n_cold} cold, {len(blocks)} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
