"""Joint H-distribution fit of several surveys: conditional shape + Poisson counts.

For survey ``s`` in filter ``f``, with population efficiency ``eps_s(H_f)``
(``survey_efficiency``), the cold LF ``lambda(H_r)`` (tapered form, H_o free)
and per-filter colour ``c_f = cbar_f + delta_f`` with intrinsic scatter
``sigma_c``::

    lambda_f(H) = int lambda(H - c) N(c; c_f, sigma_c) dc
    Lambda_s    = int lambda_f(H) eps_s(H) dH  [- overlaps with higher-priority surveys]
    ln L = sum_s [ sum_k ln f_k(lambda_f) + N_s ln Lambda_s - Lambda_s ]
           + sum_f ln N(delta_f; 0, sigma_delta)

``f_k`` is each detection's conditional density (``hfit.ConditionalLikelihood``
on its catalogue H in the survey's filter, with the block's own selection);
the Poisson terms tie the surveys' counts, and so the normalisation, together.
``Lambda_s`` is marginalised over the Monte Carlo error of ``eps_s``.

A complete survey (``kind = "complete"``, e.g. the MPC catalogue below its
H limit) has ``eps_s = f_cut 1(H_f < h_lim)`` and the conditional density
``f_k = lambda_f(H_k) / int^{h_lim} lambda_f dH``, with the colour spread
widened by the catalogue's H scatter: ``sigma^2 = sigma_c^2 + sigma_h^2``
(``sigma_h`` alone when the catalogue H is in the model band).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from . import hfit
from .survey_efficiency import EfficiencyTable

LN10 = math.log(10.0)
SIGMA_DELTA = 0.05
SIGMA_C = 0.2
COLOUR_NODES = 11
MC_NODES = 7
ALPHA_BOUNDS = (0.001, 2.0)
# Profile grid in alpha: fine where the faint-end slope is measured, coarse above.
PROFILE_ALPHAS = np.concatenate([np.linspace(0.005, 0.8, 33), [1.0, 1.3, 1.7]])
SHAPE_NAMES = ("alpha", "beta", "h_b")


def _gh(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    return math.sqrt(2.0) * x, w / math.sqrt(math.pi)


@dataclass
class CompleteSample:
    """The sample of a complete survey: every object with catalogue H_f < h_lim."""

    names: np.ndarray
    h: np.ndarray

    def __post_init__(self):
        self.index = np.arange(len(self.h))
        self.selections = {}

    @property
    def n(self) -> int:
        return len(self.h)

    @property
    def sample(self) -> "CompleteSample":
        return self


@dataclass
class SurveyTerm:
    """One survey in the joint fit."""

    name: str
    filter: str
    like: hfit.ConditionalLikelihood | CompleteSample
    eps: EfficiencyTable
    colour: float                       # cbar_f = filter - model band (cold mean)
    overlaps: list = field(default_factory=list)   # (higher-priority survey, eps table)
    inc: np.ndarray | None = None       # detections' ecliptic inclination (for plots)
    kind: str = "survey"                # or "complete"
    h_lim: float = math.inf             # complete: catalogue H_f limit
    sigma_h: float = 0.0                # complete: catalogue H scatter [mag]

    @property
    def n(self) -> int:
        return self.like.n


@dataclass
class JointLikelihood:
    """ln L(alpha, beta, h_b, h_o, delta_f...) over several surveys."""

    surveys: list
    sigma_delta: float = SIGMA_DELTA
    sigma_c: float = SIGMA_C
    form: str = "tapered"               # or "exponential" (beta, h_b unused)
    model_band: str = "r"

    def __post_init__(self):
        self.filters = sorted({s.filter for s in self.surveys if s.filter != self.model_band})
        self.names = list(SHAPE_NAMES) + ["h_o"] + [f"delta_{f}" for f in self.filters]
        self._cx, self._cw = _gh(COLOUR_NODES)
        self._mx, self._mw = _gh(MC_NODES)
        h_all = np.concatenate([s.like.sample.h - s.colour for s in self.surveys])
        self.h_r_range = (float(h_all.min()), float(h_all.max()))

    # -- parameters ----------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.names)

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        shape = theta[:3]
        h_o = theta[3]
        delta = dict(zip(self.filters, theta[4:]))
        return shape, h_o, delta

    def colour_of(self, term: SurveyTerm, delta: dict) -> float:
        return term.colour + delta.get(term.filter, 0.0)

    def bounds(self) -> dict:
        lo, hi = self.h_r_range
        b = dict(hfit.DEFAULT_PRIORS)
        b["alpha"] = ALPHA_BOUNDS
        b["h_b"] = (lo - 6.0, hi + 5.0)
        # H_o ~ -log10(N / K) / alpha runs away as alpha -> 0; a tight box on it
        # would cut the small-alpha side of the likelihood.
        b["h_o"] = (-1e6, 1e6)
        for f in self.filters:
            b[f"delta_{f}"] = (-6 * self.sigma_delta, 6 * self.sigma_delta)
        return b

    def in_bounds(self, theta) -> bool:
        b = self.bounds()
        if self.form == "exponential":
            names = ["alpha", "h_o"] + [f"delta_{f}" for f in self.filters]
            vals = [theta[0], theta[3]] + list(theta[4:])
        else:
            names, vals = self.names, list(theta)
        return all(b[n][0] < v < b[n][1] for n, v in zip(names, vals))

    # -- LF in a filter -------------------------------------------------------

    def log_lambda_r(self, h, shape, h_o):
        h = np.asarray(h, dtype=float)
        if self.form == "exponential":
            return hfit.log_differential_exponential(h, shape[0], h_o)
        return hfit.log_differential(h, *shape, h_o)

    def log_lambda_f(self, h, shape, h_o, colour, filt=None, sigma_h: float = 0.0):
        """ln of lambda(H - c) averaged over c ~ N(colour, sigma), sigma^2 = sigma_c^2 + sigma_h^2.

        In the model band itself (``filt == model_band``) there is no colour
        and only the extra scatter ``sigma_h``.
        """
        h = np.asarray(h, dtype=float)
        if filt == self.model_band:
            colour, sigma = 0.0, sigma_h
        else:
            sigma = math.hypot(self.sigma_c, sigma_h)
        if sigma <= 0.0:
            return self.log_lambda_r(h - colour, shape, h_o)
        shifts = colour + sigma * self._cx
        terms = np.stack([self.log_lambda_r(h - c, shape, h_o) for c in shifts])
        return logsumexp(terms + np.log(self._cw)[:, None], axis=0)

    def log_lambda_term(self, term: SurveyTerm, h, shape, h_o, colour):
        return self.log_lambda_f(h, shape, h_o, colour, term.filter, term.sigma_h)

    # -- terms ----------------------------------------------------------------

    def conditional(self, term: SurveyTerm, shape, h_o, colour) -> float:
        like = term.like
        if term.kind == "complete":
            from .survey_efficiency import step_weights

            ld = self.log_lambda_term(term, like.h, shape, h_o, colour)
            h = term.eps.h
            w = step_weights(h, term.h_lim) * np.gradient(h)
            lg = self.log_lambda_term(term, h[w > 0], shape, h_o, colour)
            norm = float(logsumexp(lg + np.log(w[w > 0])))
            if not (np.all(np.isfinite(ld)) and np.isfinite(norm)):
                return -np.inf
            return float(np.sum(ld) - like.n * norm)
        lg = self.log_lambda_f(like.grid, shape, h_o, colour, term.filter)
        ld = like._point_log_lambda(self.log_lambda_f(like.h_pt, shape, h_o, colour, term.filter))
        if ld is None or not np.any(np.isfinite(lg)):
            return -np.inf
        return like.loglike_from(lg, ld)

    def expected(self, term: SurveyTerm, shape, h_o, delta, boot: bool = False):
        """Lambda_s (minus overlaps), and its replicates over eps bootstraps."""
        colour = self.colour_of(term, delta)
        lam = np.exp(self.log_lambda_term(term, term.eps.h, shape, h_o, colour))
        dh = np.gradient(term.eps.h)
        total = float(np.sum(lam * term.eps.eps * dh))
        reps = (term.eps.boot * (lam * dh)[None, :]).sum(axis=1) if boot else None
        if term.overlaps:
            for _, table in term.overlaps:
                lam_r = np.exp(self.log_lambda_r(table.h, shape, h_o))
                d = np.gradient(table.h)
                total -= float(np.sum(lam_r * table.eps * d))
                if boot and len(table.boot):
                    reps = reps - (table.boot * (lam_r * d)[None, :]).sum(axis=1)
        return total, reps

    def poisson(self, term: SurveyTerm, shape, h_o, delta) -> float:
        lam, reps = self.expected(term, shape, h_o, delta, boot=True)
        if not lam > 0.0:
            return -np.inf
        n = term.n
        s = float(np.std(reps, ddof=1) / np.mean(reps)) if reps is not None and len(reps) > 1 \
            else 0.0
        if s <= 0.0:
            return n * math.log(lam) - lam - gammaln(n + 1)
        lams = lam * np.exp(s * self._mx - 0.5 * s * s)
        logp = n * np.log(lams) - lams - gammaln(n + 1)
        return float(logsumexp(logp + np.log(self._mw)))

    def loglike(self, theta) -> float:
        if not self.in_bounds(theta):
            return -np.inf
        shape, h_o, delta = self.unpack(theta)
        total = 0.0
        for term in self.surveys:
            colour = self.colour_of(term, delta)
            c = self.conditional(term, shape, h_o, colour)
            if not np.isfinite(c):
                return -np.inf
            p = self.poisson(term, shape, h_o, delta)
            if not np.isfinite(p):
                return -np.inf
            total += c + p
        for f in self.filters:
            total += -0.5 * (delta[f] / self.sigma_delta) ** 2
        return float(total)

    # -- H_o at fixed shape -----------------------------------------------------

    def best_h_o(self, shape, delta) -> float:
        """H_o maximising the Poisson terms (ignoring eps errors): A = sum N / sum K."""
        k = sum(self.expected(t, shape, 0.0, delta)[0] for t in self.surveys)
        n = sum(t.n for t in self.surveys)
        if not k > 0.0:
            return float("nan")
        return -math.log10(n / k) / shape[0]

    def detected_density(self, term: SurveyTerm, theta) -> tuple[np.ndarray, np.ndarray]:
        """(H_f, lambda_f eps_s): the model's distribution of detected H in survey s."""
        shape, h_o, delta = self.unpack(theta)
        lam = np.exp(self.log_lambda_term(term, term.eps.h, shape, h_o,
                                          self.colour_of(term, delta)))
        return term.eps.h, lam * term.eps.eps


# ---------------------------------------------------------------------------
# Fitting


@dataclass
class JointFit:
    theta: np.ndarray
    loglike: float
    profile_alpha: np.ndarray
    profile_loglike: np.ndarray
    profile_theta: np.ndarray


def _nm(fun, x0, scale=None, maxiter=6000):
    from scipy import optimize

    x0 = np.asarray(x0, dtype=float)
    simplex = None
    if scale is not None:
        simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[k] * scale[k] for k in range(len(x0))])
    return optimize.minimize(fun, x0, method="Nelder-Mead",
                             options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": maxiter,
                                      "initial_simplex": simplex})


def _with_h_o(like: JointLikelihood, shape, delta_vec):
    delta = dict(zip(like.filters, delta_vec))
    h_o = like.best_h_o(shape, delta)
    return np.concatenate([shape, [h_o], delta_vec])


def fit_joint(like: JointLikelihood, alphas=PROFILE_ALPHAS) -> JointFit:
    """MLE: grid starts over the shape (H_o profiled), a profile in alpha, then a polish."""
    nd = len(like.filters)
    lo, hi = like.h_r_range
    exponential = like.form == "exponential"

    def neg_free(x, alpha=None):
        if alpha is not None:
            x = np.concatenate([[alpha], x])
        if exponential:
            shape = np.array([x[0], 1.0, 0.0])
            delta_vec = np.asarray(x[1:])
        else:
            shape = np.asarray(x[:3])
            delta_vec = np.asarray(x[3:])
        theta = _with_h_o(like, shape, delta_vec)
        if not np.all(np.isfinite(theta)):
            return 1e300
        v = like.loglike(theta)
        return -v if np.isfinite(v) else 1e300

    if exponential:
        starts = [np.concatenate([[a], np.zeros(nd)]) for a in (0.2, 0.4, 0.6, 0.9)]
        scale = [0.05] + [0.02] * nd
    else:
        starts = [np.concatenate([[a, b, hb], np.zeros(nd)])
                  for a in (0.05, 0.2, 0.4, 0.7) for b in (0.2, 0.6, 1.5)
                  for hb in (lo, 0.5 * (lo + hi), hi)]
        scale = [0.05, 0.1, 0.5] + [0.02] * nd
    scored = sorted(starts, key=neg_free)[:6]
    best = min((_nm(neg_free, s, scale) for s in scored), key=lambda r: r.fun)

    a_lo, a_hi = like.bounds()["alpha"]
    alphas = np.asarray([a for a in alphas if a_lo < a < a_hi], dtype=float)
    prof_ll = np.full(len(alphas), -np.inf)
    prof_th = np.zeros((len(alphas), like.ndim))
    if not exponential:
        warm = None
        fixed = [np.concatenate([[b, hb], np.zeros(nd)]) for b in (0.3, 1.9) for hb in (lo, hi)]
        for k, a in enumerate(alphas):
            screened = min(fixed, key=lambda x: neg_free(x, a))
            runs = [screened] + ([warm] if warm is not None else [])
            res = min((_nm(lambda x: neg_free(x, a), c, scale[1:], maxiter=1500) for c in runs),
                      key=lambda r: r.fun)
            warm = res.x
            prof_ll[k] = -res.fun
            prof_th[k] = _with_h_o(like, np.concatenate([[a], res.x[:2]]), res.x[2:])
        k = int(np.argmax(prof_ll))
        res = _nm(neg_free, np.concatenate([[alphas[k]], prof_th[k, 1:3], prof_th[k, 4:]]), scale)
        if res.fun < best.fun:
            best = res
    x = best.x
    if exponential:
        theta = _with_h_o(like, np.array([x[0], 1.0, 0.0]), np.asarray(x[1:]))
    else:
        theta = _with_h_o(like, np.asarray(x[:3]), np.asarray(x[3:]))
    # polish with H_o free
    if exponential:
        idx = [0, 3] + list(range(4, like.ndim))
    else:
        idx = list(range(like.ndim))

    def neg_all(y):
        t = theta.copy()
        t[idx] = y
        v = like.loglike(t)
        return -v if np.isfinite(v) else 1e300

    res = _nm(neg_all, theta[idx])
    if res.fun < -like.loglike(theta):
        theta[idx] = res.x
    return JointFit(theta, like.loglike(theta), alphas, prof_ll, prof_th)


def run_mcmc(like: JointLikelihood, start, n_walkers=32, n_steps=3000, burn=1000, seed=42):
    import emcee

    rng = np.random.default_rng(seed)
    if like.form == "exponential":
        free = [0, 3] + list(range(4, like.ndim))
    else:
        free = list(range(like.ndim))
    ndim = len(free)
    scale = np.array([0.02, 0.05, 0.1, 0.05] + [0.01] * len(like.filters))[free]
    base = np.asarray(start, dtype=float)

    def logp(y):
        t = base.copy()
        t[free] = y
        return like.loglike(t)

    p0 = []
    for _ in range(1000 * n_walkers):
        if len(p0) == n_walkers:
            break
        trial = base[free] + scale * rng.standard_normal(ndim)
        if np.isfinite(logp(trial)):
            p0.append(trial)
    else:
        raise ValueError("joint MCMC: ln L not finite near the start")
    sampler = emcee.EnsembleSampler(n_walkers, ndim, logp)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(np.array(p0), n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    out = np.tile(base, (len(chain), 1))
    out[:, free] = chain
    return out, sampler.get_log_prob(discard=burn, flat=True), float(np.mean(sampler.acceptance_fraction))
