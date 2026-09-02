"""DSP-processed waveforms via dspeed's :class:`WaveformBrowser`.

The browser already builds the per-detector processing chain (resolving the
``db.*`` parameters from the dataprod config) and exposes the intermediate
waveforms as matplotlib lines; we reuse it purely as a processing engine and
pull the ``(x, y)`` data out to draw in Bokeh.

Browsers are expensive to build, so one is cached per ``(raw_file, channel)``
and reused across events via :meth:`WaveformBrowser.find_entry`.

dspeed (and the matplotlib it drags in) is imported lazily -- roughly half a
second of the worker's startup -- because most sessions only ever look at raw
waveforms and never reach this module's one dspeed call site.
"""

from __future__ import annotations

#: Intermediate processing-chain waveforms offered by the dsp view.
PROCESSED_PARAMS = ("wf_blsub", "wf_pz", "wf_trap", "curr")

#: Browsers kept alive per session. Each holds open read buffers, so the cache
#: is a bounded LRU. It must comfortably exceed the largest system's channel
#: count, or the all-waveforms view evicts every browser it just built on each
#: update; browsers for other files are dropped eagerly instead (see
#: ``_browser``), so the bound is what one file's channels need, not a budget
#: spread across every file the session has visited.
MAX_CACHED_BROWSERS = 256


def _waveform_browser():
    """Import dspeed's browser on first use, headless."""
    import matplotlib as mpl  # noqa: PLC0415 (lazy: ~0.5 s off worker startup)

    # WaveformBrowser draws onto matplotlib axes internally; keep it headless.
    mpl.use("Agg")
    from dspeed.vis import WaveformBrowser  # noqa: PLC0415

    return WaveformBrowser


class WaveformProcessor:
    """Lazily build/cache dspeed browsers and return processed traces."""

    def __init__(self, ev):
        self.ev = ev
        self._browsers: dict = {}  # (raw_file, channel) -> WaveformBrowser (LRU)
        self._current_file = None  # raw file the cached browsers belong to
        self._dsp_cfgs: dict = {}  # tstamp -> {detector_name: dsp_config_path}

    def _dsp_config(self, name):
        ts = self.ev.tstamp
        if ts not in self._dsp_cfgs:
            cfg = self.ev.meta.dataprod.config.on(ts)
            self._dsp_cfgs[ts] = cfg["snakemake_rules"]["tier_dsp"]["inputs"][
                "processing_chain"
            ]
        return self._dsp_cfgs[ts][name]

    def _browser(self, rawid, name):
        raw_file = str(self.ev.raw_file())
        channel = f"ch{rawid:07d}"
        key = (raw_file, channel)
        if raw_file != self._current_file:
            # moved to another run/file: those browsers can never be reused,
            # and holding them would push this file's own out of the cache
            for stale in [k for k in self._browsers if k[0] != raw_file]:
                self._browsers.pop(stale)
            self._current_file = raw_file
        browser = self._browsers.pop(key, None)  # re-insert below (LRU order)
        if browser is None:
            browser = _waveform_browser()(
                raw_file,
                f"{channel}/raw",
                dsp_config=self._dsp_config(name),
                lines=list(PROCESSED_PARAMS),
                buffer_len=1,
            )
        self._browsers[key] = browser
        while len(self._browsers) > MAX_CACHED_BROWSERS:
            self._browsers.pop(next(iter(self._browsers)))
        return browser

    def processed(self, rawid, name, hit_idx, param):
        """Return ``(x, y)`` arrays for ``param`` at this detector's hit row."""
        browser = self._browser(rawid, name)
        browser.find_entry(hit_idx, append=False)
        line = browser.lines[param][0]
        return line.get_xdata(), line.get_ydata()
