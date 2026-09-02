"""Guards against redundant figure rebuilds in :class:`EventDisplay`.

The expensive work behind each tab is one builder call -- ``plot_all_waveforms``
reads ~100 waveforms, ``dataset_figure`` walks every run of the cycle -- so
counting *those* (rather than the ``_update_*`` methods, which self-gate and
return early) measures the work a single user interaction actually costs. These
are the regressions that are otherwise invisible: the app stays correct, it
just does the work three times.

Needs a real production cycle; set ``$LEDS_BASE_PATH`` (the mock cycle used by
the ``run-app`` skill is enough) or these skip.
"""

from __future__ import annotations

import os

import pytest

from leds import app as app_mod

#: Expensive builders, as named in ``leds.app``'s namespace.
BUILDERS = ("plot_all_waveforms", "plot_event_waveforms")


class Counter:
    """Patches the expensive builders in ``leds.app`` and tallies real rebuilds."""

    def __init__(self, monkeypatch):
        self.counts = dict.fromkeys(BUILDERS, 0)
        for name in BUILDERS:
            monkeypatch.setattr(app_mod, name, self._wrap(name))

    def _wrap(self, name):
        inner = getattr(app_mod, name)

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return inner(*args, **kwargs)

        return counted

    def reset(self):
        for key in self.counts:
            self.counts[key] = 0


@pytest.fixture
def display():
    if not os.environ.get("LEDS_BASE_PATH"):
        pytest.skip("set $LEDS_BASE_PATH to a production cycle to run these")
    disp = app_mod.EventDisplay()
    if disp.viewer is None or not disp.period:
        pytest.skip(f"no usable cycle: {disp._cycle_error}")
    return disp


@pytest.fixture
def counter(monkeypatch):
    return Counter(monkeypatch)


def test_system_change_rebuilds_all_waveforms_once(display, counter):
    """Changing System must not fan out through the grouping/kind/category watchers."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    counter.reset()

    display.all_wf_system = "spms" if display.all_wf_system == "geds" else "geds"

    assert counter.counts["plot_all_waveforms"] == 1


def test_grouping_change_rebuilds_all_waveforms_once(display, counter):
    display.tabs.active = app_mod.TAB_WAVEFORMS
    counter.reset()

    groupings = display.param.all_wf_grouping.objects
    other = next((g for g in groupings if g != display.all_wf_grouping), None)
    if other is None:
        pytest.skip("only one grouping available for this system")
    display.all_wf_grouping = other

    assert counter.counts["plot_all_waveforms"] == 1


def test_returning_to_a_tab_does_not_rebuild(display, counter):
    """Nothing changed while we were away, so there is nothing to redo."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    display.tabs.active = app_mod.TAB_EVENT
    counter.reset()

    display.tabs.active = app_mod.TAB_WAVEFORMS

    assert counter.counts["plot_all_waveforms"] == 0


def test_new_event_rebuilds_the_active_tab_once(display, counter):
    """A new event must still refresh the tab the user is looking at -- once."""
    display.tabs.active = app_mod.TAB_WAVEFORMS
    counter.reset()

    display.index += 1

    assert counter.counts["plot_all_waveforms"] == 1


def test_inactive_tabs_are_not_rebuilt_on_a_new_event(display, counter):
    """The gate is what keeps playback from reading ~100 waveforms per step."""
    display.tabs.active = app_mod.TAB_EVENT
    counter.reset()

    display.index += 1

    assert counter.counts["plot_all_waveforms"] == 0
