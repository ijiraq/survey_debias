"""Horvitz–Thompson debiasing of pencil-beam TNO surveys with OSSSSim.

The simulator-backed runner lives in :mod:`survey_debias.grid_bias_run` and
is not imported here, so the geometry helpers and survey loading work
without importing ``ossssim``.
"""
from importlib.metadata import PackageNotFoundError, version

from .grid_bias import GridSurvey, OrbitModelCatalog, default_orbit_model_path
from .survey_config import load_survey

try:
    __version__ = version("survey_debias")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "GridSurvey",
    "OrbitModelCatalog",
    "default_orbit_model_path",
    "load_survey",
    "__version__",
]
