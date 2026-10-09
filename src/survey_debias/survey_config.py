"""Load a :class:`GridSurvey` from a TOML file or a Python object.

Accepted ``--survey`` forms:

- ``path/to/survey.toml``: ``[survey]`` holds ``GridSurvey`` quantities.
  ``[detection_columns]`` maps every program column that is read onto a
  column in the detections table, including ``mag``.
  ``[detection_defaults]`` gives ``survey``/``block``/``comp`` for a table
  without those columns.
- ``path/to/survey.py:NAME`` or ``package.module:NAME``: a ``GridSurvey``
  object defined in Python. ``:NAME`` may be omitted if the module defines
  exactly one ``GridSurvey``.

Survey geometry (pointings, epochs, observer, efficiency files) is read from
``characterization/{survey}/pointings.list``; colours from ``colour.toml``.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import tomllib
from pathlib import Path

from .grid_bias import REMOVED_FIELDS, GridSurvey

_FIELDS = {f.name: f for f in dataclasses.fields(GridSurvey)}
_REQUIRED = {
    name for name, f in _FIELDS.items()
    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
}
BIAS_METHODS = ("aq_grid", "model_ae")
_TABLES = ("survey", "detection_columns", "detection_defaults")


def survey_from_mapping(values: dict, source: str = "survey") -> GridSurvey:
    """Build and validate a ``GridSurvey`` from a plain mapping."""
    if "mag_column" in values:
        raise ValueError(
            f"{source}: mag_column is not a survey field. "
            "Name the magnitude column in [detection_columns], "
            'for example mag = "mag".'
        )
    if "column_map" in values:
        raise ValueError(
            f"{source}: column_map is not a survey field. "
            "Name every column that is read in [detection_columns]."
        )
    removed = [name for name in values if name in REMOVED_FIELDS]
    if removed:
        where = "; ".join(f"{name} -> {REMOVED_FIELDS[name]}" for name in removed)
        raise ValueError(
            f"{source}: {', '.join(removed)} no longer belong in the survey file. "
            f"Each survey's geometry is read from characterization/{{survey}}/"
            f"pointings.list and colours from colour.toml ({where})."
        )
    unknown = sorted(set(values) - set(_FIELDS))
    if unknown:
        raise ValueError(f"{source}: unknown GridSurvey field(s): {', '.join(unknown)}")
    missing = sorted(_REQUIRED - set(values))
    if missing:
        raise ValueError(f"{source}: missing required field(s): {', '.join(missing)}")
    values = dict(values)
    if "surveys" in values:
        values["surveys"] = tuple(values["surveys"])
    try:
        survey = GridSurvey(**values)
    except ValueError as ex:
        raise ValueError(f"{source}: {ex}") from ex
    if survey.bias_method not in BIAS_METHODS:
        raise ValueError(f"{source}: bias_method must be one of {BIAS_METHODS}")
    return survey


def load_survey_toml(path) -> GridSurvey:
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not any(t in data for t in _TABLES):
        return survey_from_mapping(data, source=str(path))
    extra = sorted(set(data) - set(_TABLES))
    if extra:
        raise ValueError(f"{path}: unknown table(s): {', '.join(extra)}")
    values = dict(data.get("survey", {}))
    for table in ("detection_columns", "detection_defaults"):
        if table in data:
            if table in values:
                raise ValueError(
                    f"{path}: {table} is set both under [survey] and as [{table}]"
                )
            values[table] = data[table]
    return survey_from_mapping(values, source=str(path))


def _import_module(target: str, spec_text: str):
    path = Path(target)
    if path.suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"survey module not found: {path}")
        spec = importlib.util.spec_from_file_location(f"_survey_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(target)
    except ImportError as ex:
        raise ValueError(f"cannot import survey from {spec_text!r}: {ex}") from ex


def load_survey_python(target: str, name: str | None, spec_text: str) -> GridSurvey:
    module = _import_module(target, spec_text)
    if name:
        survey = getattr(module, name, None)
        if not isinstance(survey, GridSurvey):
            raise ValueError(f"{spec_text}: {name!r} is not a GridSurvey")
        return survey
    found = [v for v in vars(module).values() if isinstance(v, GridSurvey)]
    if len(found) != 1:
        raise ValueError(
            f"{spec_text}: expected exactly one GridSurvey, found {len(found)}; "
            "use MODULE:NAME"
        )
    return found[0]


def load_survey(spec_text: str) -> GridSurvey:
    """Resolve a ``--survey`` argument to a ``GridSurvey``."""
    text = str(spec_text)
    if text.lower().endswith(".toml"):
        return load_survey_toml(text)
    target, sep, name = text.rpartition(":")
    if not sep or "/" in name or "\\" in name:
        target, name = text, ""
    return load_survey_python(target, name or None, text)
