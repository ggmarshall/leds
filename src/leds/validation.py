"""Data layer for the Validation tab: period-wide rate series + calibration.

Speed/memory design: the evt columns of one run are read once, immediately
reduced to time-binned counts (boost-histogram, 15-min base bins on a
day-aligned absolute axis) and dropped — only the tiny binned summaries are
cached (~150 kB per run), so a whole period never holds per-hit arrays.
Rebinning to the user-selected width is pure array arithmetic on the cache.

Calibration curves come from the ``par_hit`` YAMLs (per *cal* run, matched to
the viewed run via ``validity.yaml``), not from lh5 data: the fitted peak
centroids (ADC) and the calibration expression are all that is needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import awkward as ak
import boost_histogram as bh
import lh5
import numexpr as ne
import numpy as np
from dbetto import Props
from dbetto.catalog import Catalog
from lh5.io.exceptions import LH5DecodeError

#: Storage resolution of the per-run summaries. The UI bin widths below are
#: all multiples of this (and divide a day, so the day-aligned base axis
#: rebins exactly to any of them).
BASE_BIN_SECONDS = 900

#: UI label -> bin width in seconds.
BIN_WIDTHS = {
    "15 min": 900,
    "30 min": 1800,
    "1 h": 3600,
    "2 h": 7200,
    "3 h": 10800,
    "6 h": 21600,
    "12 h": 43200,
    "24 h": 86400,
}
DEFAULT_BIN_WIDTH = "1 h"

#: Potassium lines (keV) and the window (+- keV) counted around each.
K_LINES = {"K40": 1460.822, "K42": 1524.6}
LINE_WINDOW = 10.0

#: Peaks shown in the calibration-summary residual plot (keV).
PEAKS_SUMMARY = (2614.511, 583.191, 2103.511)

#: Calibration energy parameter whose curve is displayed (the production
#: energy estimator).
ENERGY_PARAM = "cuspEmax_ctc_cal"

#: Cached per-run summaries (binned counts only, KB-scale entries).
MAX_CACHED_RUNS = 64

_DAY = 86400

#: Series of each rate plot, in legend order. K-line groups are added below.
RATE_GROUPS = {
    "trigger": ("all triggers", "forced", "pulser", "muon", "muon offline"),
    "multiplicity": ("m = 0", "m = 1", "m = 2", "m > 2"),
    "qc": ("pass", "fail"),
}
KLINE_CONFIGS = ("before cuts", "after QC", "after mult = 1", "after LAr", "after PSD")
RATE_GROUPS |= {line: KLINE_CONFIGS for line in K_LINES}

#: Rate unit per group: seconds of exposure per rate unit (1 -> Hz,
#: 3600 -> counts/hour). All rates are additionally normalised by the
#: detector mass of the selected scope (one string, or the whole array).
GROUP_UNIT_SECONDS = {
    "trigger": 1,
    "multiplicity": 1,
    "qc": 1,
    **{line: 3600 for line in K_LINES},
}
GROUP_UNIT_LABEL = {1: "rate (Hz / kg)", 3600: "rate (counts / hour / kg)"}


def survival_fraction(pass_rate, fail_rate):
    """Per-bin fraction ``pass / (pass + fail)``, NaN where nothing was seen.

    The inputs are rate arrays over the same exposure, so their ratio equals
    the count ratio. Returns ``None`` if either series is unavailable.
    """
    if pass_rate is None or fail_rate is None:
        return None
    total = pass_rate + fail_rate
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, pass_rate / total, np.nan)


class ValidationData:
    """Per-run reduced rate summaries and their period-wide assembly."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._cache: dict = {}
        self._pars_cache: dict = {}
        self._strings_cache: dict = {}

    # -- per-run reduction --------------------------------------------------

    def _columns(self, period, run):
        """Read the needed evt columns of one run (not cached: reduced at once)."""
        files = [str(f) for f in self.viewer._run_files(period, run)]
        group = self.viewer.group

        def col(field):
            return lh5.read(f"{group}/{field}", files)

        def opt(field, conv):
            # tolerate cycles without a field: the series is just absent
            try:
                return conv(col(field))
            except (KeyError, LH5DecodeError, OSError):
                return None

        as_bool = lambda o: o.nda.astype(bool)  # noqa: E731
        return {
            "timestamp": col("trigger/timestamp").nda,  # required
            "forced": opt("trigger/is_forced", as_bool),
            "puls": opt("coincident/puls", as_bool),
            "muon": opt("coincident/muon", as_bool),
            "muon_offline": opt("coincident/muon_offline", as_bool),
            "spms": opt("coincident/spms", as_bool),
            "mult": opt("geds/multiplicity", lambda o: o.nda),
            "qc": opt("geds/quality/is_bb_like", as_bool),
            "rawid": opt("geds/rawid", lambda o: o.view_as("ak")),
            "energy": opt("geds/energy", lambda o: o.view_as("ak")),
            "psd_bb": opt(
                "geds/psd/is_bb_like",
                lambda o: ak.values_astype(o.view_as("ak"), bool),
            ),
        }

    @staticmethod
    def _series_masks(d, hit_sel=None):
        """Per-event boolean masks for every rate series, ``None`` if unavailable.

        All series except the raw trigger components exclude forced-trigger and
        pulser events (they are not physics); the K-line configs additionally
        apply one cut each on top of that baseline. ``hit_sel`` (a per-hit
        boolean array, e.g. hits in one string) restricts every series to
        events with a selected hit, and the K-line windows to the selected
        hits themselves.
        """
        n = len(d["timestamp"])
        true = np.ones(n, dtype=bool)
        has_hit = None if hit_sel is None else ak.to_numpy(ak.any(hit_sel, axis=1))

        def both(a, b):
            return a & b if a is not None and b is not None else None

        def scoped(m):
            return m if has_hit is None else both(m, has_hit)

        phys = (
            ~(d["forced"] | d["puls"])
            if d["forced"] is not None and d["puls"] is not None
            else None
        )
        masks = {
            ("trigger", "all triggers"): scoped(true),
            ("trigger", "forced"): scoped(d["forced"]),
            ("trigger", "pulser"): scoped(d["puls"]),
            ("trigger", "muon"): scoped(d["muon"]),
            ("trigger", "muon offline"): scoped(d["muon_offline"]),
            ("qc", "pass"): scoped(both(phys, d["qc"])),
            ("qc", "fail"): scoped(
                both(phys, ~d["qc"] if d["qc"] is not None else None)
            ),
        }
        for label, sel in (
            ("m = 0", lambda m: m == 0),
            ("m = 1", lambda m: m == 1),
            ("m = 2", lambda m: m == 2),
            ("m > 2", lambda m: m > 2),
        ):
            cond = sel(d["mult"]) if d["mult"] is not None else None
            masks[("multiplicity", label)] = scoped(both(phys, cond))

        for line, peak in K_LINES.items():
            lo, hi = peak - LINE_WINDOW, peak + LINE_WINDOW
            if d["energy"] is None:
                in_line = in_line_psd = None
            else:
                # the in-window hit must itself be a selected (in-string) hit,
                # so these masks need no extra has_hit scoping
                e = d["energy"] if hit_sel is None else d["energy"][hit_sel]
                in_line = ak.to_numpy(ak.any((e >= lo) & (e < hi), axis=1))
                if d["psd_bb"] is None:
                    in_line_psd = None
                else:
                    psd = d["psd_bb"] if hit_sel is None else d["psd_bb"][hit_sel]
                    e_psd = e[psd]
                    in_line_psd = ak.to_numpy(
                        ak.any((e_psd >= lo) & (e_psd < hi), axis=1)
                    )
            base = both(phys, in_line)
            masks[(line, "before cuts")] = base
            masks[(line, "after QC")] = both(base, d["qc"])
            masks[(line, "after mult = 1")] = both(
                base, d["mult"] == 1 if d["mult"] is not None else None
            )
            masks[(line, "after LAr")] = both(
                base, ~d["spms"] if d["spms"] is not None else None
            )
            masks[(line, "after PSD")] = both(phys, in_line_psd)
        return masks

    def _string_map(self, period, run):
        """Per-string ged rawids and masses from the run's channelmap.

        Returns ``({string: frozenset(rawids)}, {string: mass_kg})``.
        """
        tstamps = self.viewer._run_tstamps(period, run)
        if not tstamps:
            msg = f"no data files for {period} {run}"
            raise ValueError(msg)
        cached = self._strings_cache.get(tstamps[0])
        if cached is None:
            chmap = self.viewer._channelmap(tstamps[0])
            geds = chmap.map("system", unique=False).geds.map("name")
            rawids: dict = {}
            masses: dict = {}
            for name in geds:
                det = geds[name]
                string = int(det.location.string)
                rawids.setdefault(string, set()).add(int(det.daq.rawid))
                masses[string] = (
                    masses.get(string, 0.0) + float(det.production.mass_in_g) / 1000
                )
            cached = ({s: frozenset(v) for s, v in rawids.items()}, masses)
            self._strings_cache = {tstamps[0]: cached}  # one channelmap is plenty
        return cached

    def _mass_kg(self, period, run, string=None):
        """Ged mass of the scope: one string, or (``None``) the whole array."""
        masses = self._string_map(period, run)[1]
        if string is None:
            return sum(masses.values())
        return masses.get(string, 0.0)

    def available_strings(self, period):
        """Sorted string numbers of the period's first readable channelmap."""
        for run in sorted(self.viewer.available_runs().get(period, {})):
            try:
                return sorted(self._string_map(period, run)[0])
            except Exception:  # no channelmap: the selector just offers "all"
                continue
        return []

    def _hit_selection(self, d, period, run, string):
        """Per-hit boolean mask marking hits in ``string`` (ak, VoV layout)."""
        if d["rawid"] is None:
            msg = "this cycle's evt tier has no geds/rawid; cannot select a string"
            raise ValueError(msg)
        rawids = self._string_map(period, run)[0].get(string, frozenset())
        flat = np.isin(
            ak.flatten(d["rawid"]).to_numpy(),
            np.fromiter(rawids, dtype=np.int64, count=len(rawids)),
        )
        return ak.unflatten(flat, ak.num(d["rawid"]))

    def _summary(self, period, run, string=None):
        """Binned counts of one run at base resolution (LRU-cached, tiny)."""
        key = (period, run, string)
        summary = self._cache.pop(key, None)  # re-insert below (LRU order)
        if summary is None:
            d = self._columns(period, run)
            t = d["timestamp"]
            if t.size == 0:
                summary = None
            else:
                hit_sel = (
                    None
                    if string is None
                    else self._hit_selection(d, period, run, string)
                )
                # day-aligned absolute axis: every BIN_WIDTHS factor divides it
                # exactly, and bins of different runs line up with each other
                t0 = math.floor(t.min() / _DAY) * _DAY
                t1 = math.ceil(t.max() / _DAY) * _DAY
                t1 = max(t1, t0 + _DAY)
                axis = bh.axis.Regular(
                    int((t1 - t0) / BASE_BIN_SECONDS), t0, t1
                )
                series = {}
                for skey, mask in self._series_masks(d, hit_sel).items():
                    if mask is None:
                        series[skey] = None
                        continue
                    h = bh.Histogram(axis)
                    h.fill(t[mask])
                    series[skey] = h
                # seconds of data coverage per bin (clipped overlap with the
                # run's [first, last] timestamp); gaps between the run's DAQ
                # cycles are not subtracted -- rates average over them
                edges = axis.edges
                exposure = bh.Histogram(axis, storage=bh.storage.Double())
                exposure.view()[:] = np.clip(
                    np.minimum(edges[1:], t.max()) - np.maximum(edges[:-1], t.min()),
                    0.0,
                    None,
                )
                summary = {"series": series, "exposure": exposure}
        self._cache[key] = summary
        while len(self._cache) > MAX_CACHED_RUNS:
            self._cache.pop(next(iter(self._cache)))
        return summary

    # -- period assembly ------------------------------------------------------

    def period_series(self, period, bin_seconds, string=None):
        """Mass-normalised rates of every series over all runs of ``period``.

        Returns ``(times_ms, rates)`` where ``times_ms`` are bin centres in
        ms-epoch (Bokeh datetime axis) and ``rates`` maps ``(group, label)`` to
        an array aligned with ``times_ms`` (``None`` when the underlying field
        is absent in every run). Runs are separated by a NaN row so lines
        break across gaps. ``string`` restricts events to hits in that string;
        rates are divided by the scope's ged mass (per run, from its
        channelmap). Rebinned from the cached base-resolution summaries; no
        file is re-read when ``bin_seconds`` changes.
        """
        factor = bin_seconds // BASE_BIN_SECONDS
        all_keys = [(g, lbl) for g, labels in RATE_GROUPS.items() for lbl in labels]
        times: list[np.ndarray] = []
        chunks: dict = {k: [] for k in all_keys}

        for run in sorted(self.viewer.available_runs().get(period, {})):
            summary = self._summary(period, run, string)
            if summary is None:
                continue
            mass = self._mass_kg(period, run, string)
            mass = mass if mass > 0 else np.nan
            exp = summary["exposure"][:: bh.rebin(factor)]
            exp_v = exp.view()
            covered = np.flatnonzero(exp_v > 0)
            if covered.size == 0:
                continue
            sl = slice(covered[0], covered[-1] + 1)
            centers = exp.axes[0].centers[sl]
            exp_v = exp_v[sl]
            if times:  # NaN separator between runs
                for k in all_keys:
                    chunks[k].append(np.array([np.nan]))
                times.append(np.array([(centers[0] - bin_seconds) * 1000.0]))
            times.append(centers * 1000.0)
            for k in all_keys:
                h = summary["series"][k]
                if h is None:
                    chunks[k].append(np.full(exp_v.size, np.nan))
                    continue
                counts = h[:: bh.rebin(factor)].view()[sl].astype(float)
                unit = GROUP_UNIT_SECONDS[k[0]]
                chunks[k].append(counts / exp_v * unit / mass)

        if not times:
            return np.array([]), dict.fromkeys(all_keys)
        rates = {
            k: None if all(np.isnan(c).all() for c in parts) else np.concatenate(parts)
            for k, parts in chunks.items()
        }
        return np.concatenate(times), rates

    # -- calibration parameters ------------------------------------------------

    def load_cal_pars(self, period, run):
        """The ``par_hit`` calibration dict applying to ``(period, run)``.

        Returns ``(pars, source_label)``, or ``(None, reason)`` when no par
        file can be found. Resolution: the run's own cal par file if present,
        else the ``validity.yaml`` entry valid at the run's start key.
        """
        par_root = self.viewer.paths.get("par_hit")
        if not par_root:
            return None, "this cycle's dataflow config has no par_hit path"
        par_root = Path(par_root)

        par_file = None
        direct = sorted((par_root / "cal" / period / run).glob("*-par_hit.yaml"))
        if direct:
            par_file = direct[0]
        else:
            validity = par_root / "validity.yaml"
            start_key = self._start_key(period, run)
            if validity.is_file() and start_key:
                try:
                    entries = Catalog.read_from(str(validity)).valid_for(
                        start_key, allow_none=True
                    )
                except ValueError:
                    entries = None
                for entry in entries or []:
                    if str(entry).endswith("par_hit.yaml"):
                        par_file = par_root / entry
                        break
        if par_file is None or not par_file.is_file():
            return None, f"no par_hit calibration file found for {period} {run}"

        cache_key = str(par_file)
        pars = self._pars_cache.pop(cache_key, None)
        if pars is None:
            pars = Props.read_from(str(par_file))
        self._pars_cache[cache_key] = pars
        while len(self._pars_cache) > 2:  # parsed files can be MB-scale
            self._pars_cache.pop(next(iter(self._pars_cache)))

        # l200-p15-r005-cal-<ts>-par_hit.yaml -> "cal pars: p15 r005"
        parts = par_file.name.split("-")
        label = f"cal pars: {parts[1]} {parts[2]}" if len(parts) > 2 else par_file.name
        return pars, label

    def _start_key(self, period, run):
        runinfo = self.viewer.status_db.runinfo
        entry = runinfo.get(period, {}).get(run, {}).get("phy")
        if entry and entry.get("start_key"):
            return entry["start_key"]
        tstamps = self.viewer._run_tstamps(period, run)
        return tstamps[0] if tstamps else None


# -- calibration curve math (pure functions over a parsed par_hit dict) --------


def _eval_cal(operation, x):
    """Apply a par_hit calibration ``operation`` to ADC value(s) ``x``."""
    var = ENERGY_PARAM.removesuffix("_cal")
    out = ne.evaluate(
        operation["expression"],
        local_dict={var: np.asarray(x, dtype=float), **operation["parameters"]},
    )
    return np.asarray(out, dtype=float)


def cal_curve(pars, detector):
    """Calibration-curve data for one detector's detail plot.

    Returns a dict of aligned arrays: fitted peak centroids ``mu``/``mu_err``
    (ADC), their true energies ``peaks`` (keV), the calibrated positions
    ``cal_mu`` with propagated ``cal_err``, residuals, and a dense
    ``(line_x, line_y)`` sampling of the calibration expression.
    Raises ``KeyError`` if the detector has no usable ecal results.
    """
    det = pars[detector]
    operation = det["pars"]["operations"][ENERGY_PARAM]
    pk_fits = det["results"]["ecal"][ENERGY_PARAM]["pk_fits"]

    peaks, mus, mu_errs = [], [], []
    for peak_key, fit in sorted(pk_fits.items(), key=lambda kv: float(kv[0])):
        peak = float(peak_key)
        mu = fit.get("parameters", {}).get("mu")
        if not fit.get("validity") or mu is None or not np.isfinite(mu):
            continue
        peaks.append(peak)
        mus.append(float(mu))
        mu_errs.append(float(fit.get("uncertainties", {}).get("mu", np.nan)))
    if not peaks:
        msg = f"no valid peak fits for {detector}"
        raise KeyError(msg)

    peaks_arr = np.array(peaks)
    mus_arr = np.array(mus)
    mu_errs_arr = np.array(mu_errs)
    cal_mu = _eval_cal(operation, mus_arr)
    cal_err = np.abs(_eval_cal(operation, mus_arr + mu_errs_arr) - cal_mu)
    line_x = np.linspace(0.0, 1.1 * mus_arr.max(), 200)
    return {
        "peaks": peaks_arr,
        "mu": mus_arr,
        "mu_err": mu_errs_arr,
        "cal_mu": cal_mu,
        "cal_err": cal_err,
        "residual": cal_mu - peaks_arr,
        "line_x": line_x,
        "line_y": _eval_cal(operation, line_x),
        "expression": operation["expression"],
    }


def cal_residuals(pars, ordered_names):
    """Per-detector calibration residuals for the summary plot.

    ``ordered_names`` is the preferred (string/position) detector order; only
    detectors present in ``pars`` are kept, and par-file detectors missing
    from the ordering are appended. Returns ``(names, residuals)`` with
    ``residuals[peak] = (res, err)`` arrays aligned to ``names`` (NaN where a
    detector lacks a valid fit for that peak).
    """
    names = [n for n in ordered_names if n in pars]
    names += sorted(set(pars) - set(names))
    residuals = {
        peak: (np.full(len(names), np.nan), np.full(len(names), np.nan))
        for peak in PEAKS_SUMMARY
    }
    kept = []
    for name in names:
        try:
            curve = cal_curve(pars, name)
        except (KeyError, TypeError):
            continue
        kept.append(name)
        i = len(kept) - 1
        for peak in PEAKS_SUMMARY:
            (j,) = np.where(np.isclose(curve["peaks"], peak))
            if j.size:
                residuals[peak][0][i] = curve["residual"][j[0]]
                residuals[peak][1][i] = curve["cal_err"][j[0]]
    for peak in PEAKS_SUMMARY:
        residuals[peak] = (
            residuals[peak][0][: len(kept)],
            residuals[peak][1][: len(kept)],
        )
    return kept, residuals
