"""The accumulating spectrum's fast paths must agree with the honest one.

The playback view re-histograms on every frame, so it takes two shortcuts: a
prefix of the run's flattened hits instead of a per-event mask, and adding one
event's hits to a running total instead of redoing the run. Both must give
exactly what the general cut-and-flatten path gives.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from leds.spectrum import ENERGY_RANGE, N_BINS, RunSpectrum

# per-event hit energies, including empty events and a high-multiplicity one
HITS = [
    [1000.0, 2614.5],
    [],
    [583.2],
    [100.0, 200.0, 300.0, 400.0],
    [2103.5],
    [],
    [1460.8, 1460.8],
]


class FakeViewer:
    group = "evt"
    cycle_key = "fake-spectrum"

    def _run_files(self, _period, _run):
        return []


@pytest.fixture
def spectrum(monkeypatch):
    """A RunSpectrum whose run data is the synthetic ``HITS`` above."""
    energy = ak.Array(HITS)
    n = len(HITS)
    data = {
        "energy": energy,
        "flat": np.asarray(ak.to_numpy(ak.flatten(energy))),
        "hit_ends": np.cumsum(np.asarray(ak.to_numpy(ak.num(energy)))),
        "psd_bb": None,
        "puls": None,
        "muon": None,
        "spms": None,
        "forced": None,
        "qc": None,
        "mult": np.ones(n, dtype=np.uint16),
    }
    spec = RunSpectrum(FakeViewer())
    monkeypatch.setattr(spec, "_load", lambda _p, _r: data)
    return spec


def full_reference(upto_index):
    """The obvious, slow answer: histogram the hits of events 0..upto_index."""
    values = np.array([e for hits in HITS[: upto_index + 1] for e in hits])
    return np.histogram(values, bins=N_BINS, range=ENERGY_RANGE)[0]


@pytest.mark.parametrize("upto", range(len(HITS)))
def test_prefix_path_matches_the_full_computation(spectrum, upto):
    counts, _edges = spectrum.histogram("p01", "r001", upto_index=upto)

    assert np.array_equal(counts, full_reference(upto))


def test_incremental_steps_match_recomputing_from_scratch(spectrum):
    """Walk the run one event at a time, as playback does."""
    running = np.zeros(N_BINS, dtype=int)

    for event in range(len(HITS)):
        new = spectrum.hits_between("p01", "r001", event, event)
        running = running + np.histogram(new, bins=N_BINS, range=ENERGY_RANGE)[0]
        assert np.array_equal(running, full_reference(event)), f"after event {event}"


def test_hits_upto_is_a_prefix_of_the_run(spectrum):
    assert list(spectrum.hits_upto("p01", "r001", 0)) == [1000.0, 2614.5]
    assert list(spectrum.hits_upto("p01", "r001", 1)) == [1000.0, 2614.5]
    assert list(spectrum.hits_upto("p01", "r001", 2)) == [1000.0, 2614.5, 583.2]


def test_index_beyond_the_run_is_clamped(spectrum):
    counts, _edges = spectrum.histogram("p01", "r001", upto_index=10_000)

    assert np.array_equal(counts, full_reference(len(HITS) - 1))


def test_cuts_still_take_the_general_path(spectrum):
    """A cut must not be silently ignored by the uncut fast path."""
    counts, _edges = spectrum.histogram(
        "p01", "r001", upto_index=len(HITS) - 1, cuts={"multiplicity": ">2"}
    )

    # only the 4-hit event survives a >2 multiplicity cut... except `mult` is
    # all ones here, so nothing does
    assert counts.sum() == 0
