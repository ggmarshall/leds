"""Bokeh "all waveforms" view, per detector system.

``system`` selects the detector system (geds / spms). ``grouping`` partitions
that system (see ``leds.event_viewer.SYSTEM_GROUPINGS``: geds by
String/HV filter/CC4, spms by barrel). ``category`` then picks the subset:

* ``"all"`` — every channel of the system,
* ``"above threshold"`` — geds channels above ``ABOVE_THRESHOLD`` (geds only),
* a specific group label (e.g. ``"String:01"``, ``"IB"``).

Views: *compressed* (one figure, traces overlaid + per-group/detector legend
with ``click_policy="hide"``) or *exploded* (a grid of subplots).

Traces are a raw waveform of the system's ``kind`` (geds:
waveform_windowed/presummed, spms: waveform_bit_drop), optionally
baseline-subtracted; for geds only, ``param`` may instead be a dsp-processed
waveform via a :class:`~leds.waveform_proc.WaveformProcessor`. Waveforms are
read at the event index (physics-mode tables are event-aligned).

The figure is held in an :class:`AllWaveformsFigure` and reused across events:
its layout (figure or grid, renderers, legend, which detectors) is built once
and later events only replace the source columns, and every trace is
decimated to :data:`MAX_TRACE_POINTS` before it is sent to the browser.
"""

from __future__ import annotations

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Category20
from bokeh.plotting import figure
from lh5.io.exceptions import LH5DecodeError

from leds.event_viewer import SYSTEM_GROUPINGS, group_channels
from leds.waveform_view import RAW

ABOVE_THRESHOLD = 25.0  # keV
_READ_ERRORS = (OSError, KeyError, IndexError, LH5DecodeError)
SYSTEMS = tuple(SYSTEM_GROUPINGS)  # ("geds", "spms")
#: raw waveform kinds offered per system (geds also supports dsp via `param`)
SYSTEM_KINDS = {
    "geds": ("waveform_windowed", "waveform_presummed"),
    "spms": ("waveform_bit_drop",),
}

#: Samples kept per trace on the wire. The traces are drawn a thousand or so
#: pixels wide, so keeping each bucket's min and max loses nothing visible
#: while a 6250-sample geds waveform shrinks ~4x -- with ~60 of them per
#: render that is what a step on this tab costs over NERSC's ingress. The
#: Event tab's own waveform panel stays at full resolution for inspecting
#: individual samples.
MAX_TRACE_POINTS = 1500


def decimate(x, y, max_points=MAX_TRACE_POINTS):
    """Downsample ``(x, y)`` to at most ``max_points``, keeping every extreme.

    Consecutive samples are grouped into ``max_points // 2`` buckets and each
    bucket's minimum and maximum are kept, in sample order, so pulse peaks and
    baselines survive where a plain stride would alias them. Traces already
    within the budget are returned unchanged.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)
    buckets = max_points // 2
    if n <= max_points or buckets < 1:
        return x, y
    width = -(-n // buckets)  # ceil(n / buckets)
    pad = buckets * width - n
    # pad with the last sample: it already sits in the last bucket, so that
    # bucket's extremes are unchanged
    y_padded = np.concatenate([y, np.full(pad, y[-1], dtype=y.dtype)]) if pad else y
    blocks = y_padded.reshape(buckets, width)
    offsets = np.arange(buckets) * width
    lows = blocks.argmin(axis=1) + offsets
    highs = blocks.argmax(axis=1) + offsets
    # min and max of each bucket in time order, then clip any index that
    # landed on the padding back onto the (identical) last real sample
    idx = np.sort(np.stack([lows, highs], axis=1), axis=1).ravel()
    idx = np.minimum(idx, n - 1)
    return x[idx], y[idx]


def groupings_for(system):
    return list(SYSTEM_GROUPINGS[system])


def raw_kinds_for(system):
    return list(SYSTEM_KINDS[system])


def category_options(ev, system, grouping):
    # cached per channelmap: recomputed on every event render otherwise
    key = ("wf_categories", ev.tstamp, system, grouping)
    options = ev.view_cache.get(key)
    if options is None:
        groups = list(group_channels(ev.chmap, system, grouping))
        # "above threshold" uses geds hit energies, so only for geds
        base = ["all", "above threshold"] if system == "geds" else ["all"]
        options = base + groups
        ev.view_cache[key] = options
    return options


_MAX_LEGEND_ROWS = 16  # wrap into more columns beyond this many items


def _color(i):
    return Category20[20][i % 20]


def _legend(items, font_size="8pt"):
    ncols = max(1, -(-len(items) // _MAX_LEGEND_ROWS))  # ceil(n / max_rows)
    return Legend(
        items=items,
        click_policy="hide",
        label_text_font_size=font_size,
        ncols=ncols,
    )


def _trace(ev, rawid, name, param, processor, subtract_baseline, kind):
    if param == RAW:
        x, y = ev.read_waveform(rawid, ev.index, kind)
        y = y.astype(float)
        if subtract_baseline:
            y = y - y[:100].mean()
    else:
        x, y = processor.processed(rawid, name, ev.index, param)
    return decimate(x, y)


def y_axis_label(param, subtract_baseline):
    if param == RAW:
        return "amplitude (ADC)" + (
            ", baseline-subtracted" if subtract_baseline else ""
        )
    return param


def _detectors(ev, system, grouping, category):
    """Ordered ``[(rawid, name, group_label)]`` for the category."""
    groups = group_channels(ev.chmap, system, grouping)
    cmap = ev.chmap.map("daq.rawid")
    if category == "above threshold":
        fired = sorted(ev.fired_detectors, key=lambda d: d["energy"], reverse=True)
        return [
            (d["rawid"], d["name"], None)
            for d in fired
            if d["energy"] > ABOVE_THRESHOLD
        ]
    if category in groups:
        return [(r, cmap[r]["name"], category) for r in groups[category]]
    return [
        (r, cmap[r]["name"], label) for label, rawids in groups.items() for r in rawids
    ]


def _new_figure(param, subtract_baseline, **kwargs):
    return figure(
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        x_axis_label="time (ns)",
        y_axis_label=y_axis_label(param, subtract_baseline),
        **kwargs,
    )


def _ncols(n):
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    if n <= 9:
        return 3
    return 4


def _groups(ev, system, grouping, category):
    """``[(title, [(rawid, name), ...])]`` — one entry per exploded subplot."""
    groups = group_channels(ev.chmap, system, grouping)
    cmap = ev.chmap.map("daq.rawid")
    if category == "above threshold":
        fired = sorted(ev.fired_detectors, key=lambda d: d["energy"], reverse=True)
        return [
            (f"{d['name']} ({d['energy']:.0f} keV)", [(d["rawid"], d["name"])])
            for d in fired
            if d["energy"] > ABOVE_THRESHOLD
        ]
    if category in groups:
        return [(cmap[r]["name"], [(r, cmap[r]["name"])]) for r in groups[category]]
    return [
        (label, [(r, cmap[r]["name"]) for r in rawids])
        for label, rawids in groups.items()
    ]


class AllWaveformsFigure:
    """A reusable all-waveforms figure: build the layout once, refresh the data.

    Replacing the pane's figure on every event made the browser tear down and
    rebuild every renderer and legend (~60 of them for the geds system),
    forget its zoom, and receive the whole model tree again. The layout --
    figure or grid, renderers, legend, the detectors drawn -- depends only on
    the channelmap and the view controls, so it is built once per
    ``layout_key`` and each later event just replaces the ColumnDataSource
    columns, the same way the array view swaps its energies.
    """

    def __init__(self):
        self.layout_key = None
        #: the Bokeh model to display; a new object whenever the layout is rebuilt
        self.root = None
        # [(source, [(rawid, name), ...], is_multi_line)]
        self._traces: list = []
        # [(figure, detector name)] whose title carries the (per-event) hit energy
        self._titled: list = []

    def update(
        self,
        ev,
        *,
        system="geds",
        grouping="String",
        category="all",
        exploded=False,
        param=RAW,
        processor=None,
        subtract_baseline=True,
        kind="waveform_windowed",
        apply=None,
    ):
        """Bring the figure up to date with ``ev``'s current event.

        Returns True when the layout was rebuilt -- ``root`` is then a new model
        the caller has to display -- and False when only the data changed.

        The waveforms are read first; the resulting model writes are then
        handed to ``apply`` (default: run immediately), so a caller on a
        thread can route them onto the document's event loop.
        """
        if system != "geds":
            param = RAW  # only geds has a dsp processing chain
        dets = _detectors(ev, system, grouping, category)
        key = (
            ev.tstamp,
            system,
            grouping,
            category,
            exploded,
            param,
            subtract_baseline,
            kind,
            tuple(rawid for rawid, _name, _group in dets),
        )
        rebuilt = key != self.layout_key
        if rebuilt:
            self._traces, self._titled = [], []
            self.root = self._build(
                ev, dets, system, grouping, category, exploded, param, subtract_baseline
            )
            self.layout_key = key
        self._refresh(ev, param, processor, subtract_baseline, kind, apply)
        return rebuilt

    # -- layout ---------------------------------------------------------------

    def _build(
        self, ev, dets, system, grouping, category, exploded, param, subtract_baseline
    ):
        build = self._build_exploded if exploded else self._build_compressed
        return build(ev, dets, system, grouping, category, param, subtract_baseline)

    def _add_line(self, fig, rawid, name, **style):
        """One line renderer with its own (initially empty) source."""
        source = ColumnDataSource({"x": [], "y": []})
        self._traces.append((source, [(rawid, name)], False))
        return fig.line("x", "y", source=source, **style)

    def _build_compressed(
        self, ev, dets, system, grouping, category, param, subtract_baseline
    ):
        fig = _new_figure(
            param, subtract_baseline, sizing_mode="stretch_both", title=category
        )
        items = []
        if category == "all":
            members: dict = {}
            for rawid, name, label in dets:
                members.setdefault(label, []).append((rawid, name))
            # one multi_line per group rather than one line per channel: with
            # ~100 channels that is a dozen Bokeh renderers and sources instead
            # of a hundred, which is what makes this tab sluggish in the
            # browser. The legend already grouped the per-channel lines, so
            # hiding behaves the same.
            for i, label in enumerate(group_channels(ev.chmap, system, grouping)):
                if label not in members:
                    continue
                source = ColumnDataSource({"xs": [], "ys": []})
                self._traces.append((source, members[label], True))
                renderer = fig.multi_line(
                    xs="xs",
                    ys="ys",
                    source=source,
                    color=_color(i),
                    line_width=1,
                    alpha=0.7,
                )
                items.append((label, [renderer]))
        else:
            for i, (rawid, name, _group) in enumerate(dets):
                line = self._add_line(fig, rawid, name, color=_color(i), line_width=1.5)
                items.append((name, [line]))
        if items:
            fig.add_layout(_legend(items), "right")
        return fig

    def _build_exploded(
        self, ev, _dets, system, grouping, category, param, subtract_baseline
    ):
        # the exploded grid works from _groups (one subplot per group) rather
        # than the flat detector list the compressed view draws from
        groups = _groups(ev, system, grouping, category)
        figs = []
        shared_x = shared_y = None
        for title, members in groups:
            fig = _new_figure(
                param,
                subtract_baseline,
                height=240,
                sizing_mode="stretch_width",
                title=title,
            )
            fig.yaxis.axis_label = None  # one shared label for the whole grid instead
            if shared_x is None:
                shared_x, shared_y = fig.x_range, fig.y_range
            else:
                fig.x_range, fig.y_range = shared_x, shared_y
            if category == "above threshold":
                # the title quotes the hit energy, which changes with the event
                self._titled.append((fig, members[0][1]))
            items = [
                (
                    name,
                    [self._add_line(fig, rawid, name, color=_color(j), line_width=1)],
                )
                for j, (rawid, name) in enumerate(members)
            ]
            if items:
                fig.add_layout(_legend(items, font_size="7pt"), "right")
            figs.append(fig)

        if not figs:
            return _new_figure(
                param,
                subtract_baseline,
                sizing_mode="stretch_both",
                title="no detectors",
            )
        # stack the spms barrels (IB above OB) rather than side by side
        ncols = 1 if (system == "spms" and category == "all") else _ncols(len(figs))
        return gridplot(figs, ncols=ncols, sizing_mode="stretch_both")

    # -- data -----------------------------------------------------------------

    def _refresh(self, ev, param, processor, subtract_baseline, kind, apply=None):
        """Read this event's traces, then write them into the existing sources."""
        updates = []
        for source, members, is_multi in self._traces:
            traces = []
            for rawid, name in members:
                try:
                    x, y = _trace(
                        ev, rawid, name, param, processor, subtract_baseline, kind
                    )
                except _READ_ERRORS:
                    continue
                # float32 halves the bytes on the wire and is far finer than
                # the pixel grid these are drawn on
                traces.append(
                    (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
                )
            if is_multi:
                data = {"xs": [x for x, _y in traces], "ys": [y for _x, y in traces]}
            elif traces:
                data = {"x": traces[0][0], "y": traces[0][1]}
            else:
                # unreadable for this event: show nothing rather than the
                # previous event's trace under this event's title
                data = {"x": [], "y": []}
            updates.append((source, data))
        titles = [
            (fig, f"{name} ({(ev.energy_dict.get(name) or 0.0):.0f} keV)")
            for fig, name in self._titled
        ]

        def write():
            for source, data in updates:
                source.data = data
            for fig, text in titles:
                fig.title.text = text

        if apply is None:
            write()
        else:
            apply(write)


def plot_all_waveforms(ev, **options):
    """A fresh figure for ``ev``'s event; see :meth:`AllWaveformsFigure.update`.

    The app keeps one :class:`AllWaveformsFigure` per session and refreshes it;
    this one-shot form is for callers that just want the Bokeh model.
    """
    view = AllWaveformsFigure()
    view.update(ev, **options)
    return view.root
