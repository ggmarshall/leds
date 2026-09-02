"""geds energy spectra from the evt tier, with optional cuts.

Used by two views: the event-display tab's *accumulating* raw spectrum
(``upto_index`` set, no cuts) and the Spectrum tab's *whole-run* spectrum
(1 keV bins, cut toggles). The run's per-hit ``evt/geds/energy`` and the cut
arrays are read once and cached, so re-histogramming on a cut toggle is cheap.
"""

from __future__ import annotations

import awkward as ak
import lh5
import numpy as np
from lh5.io.exceptions import LH5DecodeError

from leds._cache import RUN_SPECTRA

#: Binary cuts: key -> (label, positive-option label, negative-option label).
#: Each option is an independent checkbox; the positive option keeps events
#: where the cut condition holds, the negative keeps where it does not. With
#: neither or both ticked the cut is off. The condition per key is in
#: ``_event_mask`` / ``_kept_energy``.
BINARY_CUTS = {
    "geds_trigger": ("geds trigger", "forced", "normal"),
    "muon": ("muon", "coincident", "anticoincident"),
    "spms": ("spms", "coincident", "anticoincident"),
    "quality": ("quality", "pass", "fail"),
    "psd": ("psd", "pass", "fail"),
}
MULT_OPTIONS = ("off", "1", "2", ">2")

#: Energy axis of the spectra.
ENERGY_RANGE = (0, 4000)
N_BINS = 1000  # ~4 keV bins (accumulating playback spectrum)
DEFAULT_BIN_WIDTH = 5.0  # keV (Spectrum tab, user-adjustable)


def bins_for_width(width):
    """Number of histogram bins over ``ENERGY_RANGE`` for a ``width`` keV bin."""
    return max(1, round((ENERGY_RANGE[1] - ENERGY_RANGE[0]) / width))


def _flatten_hits(energy):
    """``(flat_energies, hits_through_event)`` for a per-event energy VoV.

    ``ak.flatten`` returns a view of the same buffer, so this costs one
    ``int64`` per event and no copy of the energies themselves. Returns
    ``(None, None)`` for layouts that will not flatten to a plain array, which
    just disables the fast path below.
    """
    try:
        flat = np.asarray(ak.to_numpy(ak.flatten(energy)))
        hit_ends = np.cumsum(np.asarray(ak.to_numpy(ak.num(energy))))
    except Exception:
        return None, None
    return flat, hit_ends


class RunSpectrum:
    """Cut-and-histogram the geds energies of a run."""

    def __init__(self, viewer):
        self.viewer = viewer

    def _load(self, period, run):
        """The run's per-hit energies and cut arrays, shared across sessions.

        These are the largest objects the app holds -- the whole run's hits --
        so N users on the same run previously meant N copies. Sharing makes
        the second user's first spectrum instant *and* cuts the memory.
        """
        files = [str(f) for f in self.viewer._run_files(period, run)]
        group = self.viewer.group  # in-file group ("evt"), not the file tier

        def load():
            def col(field):
                return lh5.read(f"{group}/{field}", files)

            def opt(field, conv):
                # cut fields may be absent in other (non-L200) dataflow cycles;
                # a missing one just disables that cut instead of breaking the
                # whole spectrum view
                try:
                    return conv(col(field))
                except (KeyError, LH5DecodeError, OSError):
                    return None

            as_bool = lambda o: o.nda.astype(bool)  # noqa: E731
            energy = col("geds/energy").view_as("ak")  # required
            flat, hit_ends = _flatten_hits(energy)
            return {
                "energy": energy,
                # flattened view of the same buffer (no copy) plus, per event,
                # the number of hits up to and including it -- everything the
                # uncut accumulating spectrum needs, without a per-event mask
                "flat": flat,
                "hit_ends": hit_ends,
                "psd_bb": opt(
                    "geds/psd/is_bb_like",
                    lambda o: ak.values_astype(o.view_as("ak"), bool),
                ),
                "puls": opt("coincident/puls", as_bool),
                "muon": opt("coincident/muon", as_bool),
                "spms": opt("coincident/spms", as_bool),
                "forced": opt("trigger/is_forced", as_bool),
                "qc": opt("geds/quality/is_bb_like", as_bool),
                "mult": opt("geds/multiplicity", lambda o: o.nda),
            }

        key = (self.viewer.cycle_key, period, run, tuple(files))
        return RUN_SPECTRA.get(key, load)

    @staticmethod
    def _apply_binary(keep, cond, checked):
        """Tighten ``keep`` by a binary cut's (positive, negative) checkboxes."""
        pos, neg = checked
        if pos and not neg:
            keep &= cond
        elif neg and not pos:
            keep &= ~cond
        return keep  # neither or both ticked -> no filter

    @staticmethod
    def _event_mask(d, cuts):
        """Per-event boolean mask for the event-level cuts.

        Each binary cut keeps events where its condition holds (positive option:
        geds_trigger=is_forced, muon/spms=their coincidence, quality=bb-like) or
        does not (negative option). Multiplicity selects 1 / 2 / >2. A cut whose
        field is missing in this cycle (loaded as ``None``) is skipped.
        """
        keep = np.ones(len(d["energy"]), dtype=bool)
        conditions = {
            # forced = is_forced | pulser-coincident; normal = its complement
            "geds_trigger": (
                d["forced"] | d["puls"]
                if d["forced"] is not None and d["puls"] is not None
                else None
            ),
            "muon": d["muon"],
            "spms": d["spms"],
            "quality": d["qc"],
        }
        for key, cond in conditions.items():
            if cond is None:
                continue
            keep = RunSpectrum._apply_binary(keep, cond, cuts.get(key, (False, False)))

        mult = cuts.get("multiplicity", "off")
        if d["mult"] is not None:
            if mult == "1":
                keep &= d["mult"] == 1
            elif mult == "2":
                keep &= d["mult"] == 2
            elif mult == ">2":
                keep &= d["mult"] > 2
        return keep

    @staticmethod
    def _kept_energy(d, keep, cuts):
        """Per-hit energies of the kept events, with the per-hit psd cut applied."""
        energy = d["energy"][keep]
        if d["psd_bb"] is None:  # cycle without psd flags -> cut unavailable
            return energy
        pos, neg = cuts.get("psd", (False, False))
        if pos and not neg:
            energy = energy[d["psd_bb"][keep]]
        elif neg and not pos:
            energy = energy[~d["psd_bb"][keep]]
        return energy

    def _hit_end(self, d, upto_index):
        """Index one past the last hit of event ``upto_index``."""
        ends = d["hit_ends"]
        return (
            int(ends[max(0, min(int(upto_index), len(ends) - 1))]) if len(ends) else 0
        )

    def hits_upto(self, period, run, upto_index):
        """Flat energies of every hit in events ``0..upto_index`` (uncut)."""
        d = self._load(period, run)
        if d["flat"] is None:
            return None
        return d["flat"][: self._hit_end(d, upto_index)]

    def hits_between(self, period, run, first_event, last_event):
        """Flat energies of the hits in events ``first..last`` (inclusive, uncut).

        Lets the accumulating playback spectrum add one event's hits per frame
        instead of re-reducing the whole run.
        """
        d = self._load(period, run)
        if d["flat"] is None:
            return None
        start = self._hit_end(d, first_event - 1) if first_event > 0 else 0
        return d["flat"][start : self._hit_end(d, last_event)]

    def histogram(self, period, run, *, upto_index=None, cuts=None, bins=N_BINS):
        """Return ``(counts, edges)`` of the geds energies passing the cuts.

        ``upto_index`` (if given) restricts to events ``0..upto_index``;
        ``cuts`` is a ``{cut_key: bool}`` mapping (see ``CUTS``).
        """
        d = self._load(period, run)
        cuts = cuts or {}
        if upto_index is not None and not cuts and d["flat"] is not None:
            # accumulating view: with no cuts the answer is simply a prefix of
            # the run's hits, so skip the per-event mask and the re-flatten
            # that would otherwise run on every playback frame
            values = d["flat"][: self._hit_end(d, upto_index)]
            return np.histogram(values, bins=bins, range=ENERGY_RANGE)
        keep = self._event_mask(d, cuts)
        if upto_index is not None:
            keep[max(0, min(int(upto_index), len(keep) - 1)) + 1 :] = False
        values = ak.to_numpy(ak.flatten(self._kept_energy(d, keep, cuts)))
        return np.histogram(values, bins=bins, range=ENERGY_RANGE)

    def events_in_bin(self, period, run, cuts, lo, hi):
        """Run-global indices of events with a passing geds hit in ``[lo, hi)``."""
        d = self._load(period, run)
        cuts = cuts or {}
        keep = self._event_mask(d, cuts)
        energy = self._kept_energy(d, keep, cuts)
        in_bin = ak.to_numpy(ak.any((energy >= lo) & (energy < hi), axis=1))
        return np.flatnonzero(keep)[in_bin]
