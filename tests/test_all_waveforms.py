"""The all-waveforms figure is reused across events, and its traces are
decimated without hiding anything a full-resolution trace shows.

Rebuilding the figure per event is what made stepping through events on that
tab slow: the browser tore down ~60 renderers and a legend and lost its zoom,
and ~3 MB of samples crossed the websocket each time.
"""

from __future__ import annotations

import numpy as np
import pytest

from leds import all_waveforms_view as awv
from leds.all_waveforms_view import AllWaveformsFigure, decimate

# ---------------------------------------------------------------- decimate


def test_short_traces_are_returned_unchanged():
    x = np.arange(10.0)
    y = np.sin(x)

    dx, dy = decimate(x, y, max_points=20)

    assert np.array_equal(dx, x)
    assert np.array_equal(dy, y)


def test_every_bucket_extreme_survives():
    rng = np.random.default_rng(1)
    y = rng.normal(size=6250)
    y[4000] = 50.0  # a pulse peak
    y[123] = -40.0  # and a dip, both far narrower than a bucket
    x = np.arange(len(y)) * 16.0

    dx, dy = decimate(x, y, max_points=1500)

    assert len(dy) <= 1500
    assert dy.max() == y.max()
    assert dy.min() == y.min()
    assert np.all(np.diff(dx) >= 0), "still in time order"
    assert np.all(np.isin(dy, y)), "every kept point is a real sample"
    buckets = 750
    width = -(-len(y) // buckets)
    for b in range(buckets):
        seg = y[b * width : (b + 1) * width]
        if seg.size:  # the last buckets may fall entirely in the padding
            assert seg.min() in dy
            assert seg.max() in dy


def test_length_not_divisible_by_the_bucket_count():
    y = np.arange(1001.0)

    dx, dy = decimate(y, y, max_points=100)

    assert len(dy) <= 100
    assert dy[0] == 0.0
    assert dy[-1] == 1000.0
    assert np.array_equal(dx, dy)


def test_empty_trace():
    dx, dy = decimate(np.array([]), np.array([]))

    assert len(dx) == 0
    assert len(dy) == 0


# ---------------------------------------------------------------- figure reuse


class FakeChmap:
    def __init__(self, names):
        self._names = names

    def map(self, _key):
        return {rawid: {"name": name} for rawid, name in self._names.items()}


class FakeViewer:
    """Just enough of an EventViewer for the all-waveforms builders."""

    def __init__(self):
        self.tstamp = "20250101T000000Z"
        self.index = 0
        self.chmap = FakeChmap({1: "V01", 2: "V02", 3: "V03"})
        self.fired_detectors = []
        self.energy_dict = {}
        self.unreadable = set()

    def read_waveform(self, rawid, _hit_idx, _kind):
        if rawid in self.unreadable:
            msg = "channel gone"
            raise OSError(msg)
        x = np.arange(100.0) * 16.0
        # a value that identifies both the channel and the event
        return x, np.full(100, float(rawid * 10 + self.index))


GROUPS = {"String:01": [1, 2], "String:02": [3]}


@pytest.fixture(autouse=True)
def _fake_groups(monkeypatch):
    monkeypatch.setattr(
        awv, "group_channels", lambda _chmap, _system, _grouping: dict(GROUPS)
    )


def test_a_new_event_refreshes_the_same_figure():
    ev = FakeViewer()
    fig = AllWaveformsFigure()

    assert fig.update(ev, subtract_baseline=False) is True
    root = fig.root

    ev.index = 1
    assert fig.update(ev, subtract_baseline=False) is False
    assert fig.root is root, "same Bokeh model: the browser keeps its renderers"
    source, _members, is_multi = fig._traces[0]
    assert is_multi
    assert float(source.data["ys"][0][0]) == 11.0, "rawid 1 at event 1"


def test_view_controls_rebuild_the_layout():
    ev = FakeViewer()
    fig = AllWaveformsFigure()
    fig.update(ev)

    assert fig.update(ev, exploded=True) is True
    assert fig.update(ev, exploded=True, subtract_baseline=False) is True
    assert fig.update(ev, exploded=True, subtract_baseline=False) is False


def test_above_threshold_rebuilds_only_when_the_fired_set_changes():
    ev = FakeViewer()
    ev.fired_detectors = [{"rawid": 1, "name": "V01", "energy": 100.0, "hit_idx": 0}]
    ev.energy_dict = {"V01": 100.0}
    fig = AllWaveformsFigure()
    opts = {"category": "above threshold", "exploded": True}

    assert fig.update(ev, **opts) is True

    ev.index = 1
    ev.fired_detectors[0]["energy"] = 250.0
    ev.energy_dict = {"V01": 250.0}
    assert fig.update(ev, **opts) is False, "same detector set: no rebuild"
    assert "250 keV" in fig._titled[0][0].title.text, "but the title moved on"

    ev.fired_detectors.append({"rawid": 3, "name": "V03", "energy": 30.0, "hit_idx": 0})
    assert fig.update(ev, **opts) is True, "a new detector fired: rebuild"


def test_an_unreadable_channel_shows_nothing_rather_than_a_stale_trace():
    ev = FakeViewer()
    fig = AllWaveformsFigure()
    fig.update(ev, category="String:02")  # one line per detector
    source, _members, _multi = fig._traces[0]
    assert len(source.data["y"]) == 100

    ev.unreadable.add(3)
    ev.index = 1
    fig.update(ev, category="String:02")

    assert len(source.data["y"]) == 0


def test_one_shot_helper_still_returns_a_figure():
    assert awv.plot_all_waveforms(FakeViewer()) is not None
