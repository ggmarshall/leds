"""Bokeh views of the dataset metadata: data blocks, exposure, usability,
partitions, and PSD status.

The matrix views share one grid: detector rows (grouped by string, top to
bottom) x run columns (grouped by period via a nested categorical axis),
covering the runs with data in the selected production cycle. The *usability*
matrix colours each cell on/ac/off with the status-DB ``reason`` in the hover;
the *partitions* matrix colours cells by analysis partition from the
``{phy,cal}_groupings`` metadata (per-detector overrides merged over the
``default`` block); the *psd* matrix evaluates each detector's ``is_bb_like``
flag expression. The *exposure* view is per-run active-mass x livetime bars
with a cumulative line; the *data* view lays the runs out as blocks on a real
time axis (one lane per period) with livetime in the hover. Tapping any
cell/bar/block selects its ``(period, run)`` in the returned source
(click-to-jump hook for the app).

This is the interactive counterpart of the matplotlib/xlsx visualisation on
pylegendmeta's ``plotting`` branch (``legendmeta.vis``); the pure metadata
helpers (:func:`expand_run_list`, :func:`merge_with_defaults`,
:func:`partition_label`) are ported from there.

Framework-agnostic: builds Bokeh figures, no Panel.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bokeh.models import (
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LinearAxis,
    Range1d,
    Span,
    TapTool,
)
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure

from leds.event_viewer import _tstamp_to_unix

DATATYPES = ("phy", "cal")
PLOTS = ("data", "exposure", "usabilities", "partitions", "psd")

#: Status colours (same palette as legendmeta.vis).
STATUS_COLORS = {"off": "#FF4444", "ac": "#FFA500", "on": "#6BAE75"}
#: PSD flag-status colours: every flag calibrated / present-but-uncalibrated /
#: missing (same mapping as legendmeta.vis.psd).
PSD_COLORS = {
    "valid": STATUS_COLORS["on"],
    "present": STATUS_COLORS["ac"],
    "missing": STATUS_COLORS["off"],
}
UNKNOWN_COLOR = "#DDDDDD"  # present in the status DB but unrecognised value
LEGEND_BLUE = "#1A2A5B"

ROW_PX = 15  # heatmap row height per detector

SECONDS_PER_YEAR = 365.25 * 86400


# ---------------------------------------------------------------------------
# metadata helpers (ported from legendmeta.vis on the `plotting` branch)
# ---------------------------------------------------------------------------


def expand_run_list(value):
    """Expand a groupings run value (``"r000..r005"``, list, or scalar) to runs."""
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(expand_run_list(item))
        return out
    s = str(value)
    if ".." in s:
        start, end = s.split("..", 1)
        return [f"r{n:03d}" for n in range(int(start[1:]), int(end[1:]) + 1)]
    return [s]


def merge_with_defaults(det_partitions, default_partitions):
    """Merge one detector's partition overrides with the ``default`` block.

    Returns ``{(period, run): partition_name}``. A partition present in the
    detector's own block takes its periods from there; default periods it does
    not mention still apply, as do partitions only defined in the default.
    """
    result = {}
    names = set(det_partitions) | set(default_partitions)
    for part in names:
        default_source = default_partitions.get(part, {})
        if not isinstance(default_source, dict):
            default_source = {}
        det_source = det_partitions.get(part)
        if det_source is not None and not isinstance(det_source, dict):
            continue
        if det_source is not None:
            for period, runs in det_source.items():
                for run in expand_run_list(runs):
                    result[(period, run)] = part
            for period, runs in default_source.items():
                if period not in det_source:
                    for run in expand_run_list(runs):
                        result[(period, run)] = part
        else:
            for period, runs in default_source.items():
                for run in expand_run_list(runs):
                    result[(period, run)] = part
    return result


def partition_num(part_name):
    """``"phygroup008b"`` -> ``8``."""
    return int(part_name.split("group")[-1][:-1])


def partition_label(part_name):
    """``"phygroup008b"`` -> ``"8b"``."""
    tail = part_name.split("group")[-1]
    return f"{int(tail[:-1])}{tail[-1]}"


# ---------------------------------------------------------------------------
# grid assembly
# ---------------------------------------------------------------------------


def _columns(viewer, datatype):
    """Ordered run columns: ``[(period, run, start_key|None, livetime|None)]``.

    One column per run with data in the cycle; ``start_key`` is None when the
    run has no ``datatype`` entry in ``runinfo``.
    """
    runinfo = viewer.status_db.runinfo
    runs = viewer.available_runs()
    cols = []
    for period in sorted(runs):
        for run in sorted(runs[period]):
            entry = runinfo.get(period, {}).get(run, {}).get(datatype)
            cols.append(
                (
                    period,
                    run,
                    entry["start_key"] if entry else None,
                    entry.get("livetime_in_s") if entry else None,
                )
            )
    return cols


def _detector_rows(viewer, tstamp):
    """Ged names ordered by (string, position), plus each row's string number."""
    chmap = viewer._channelmap(tstamp)
    geds = chmap.map("system", unique=False).geds.map("name")
    rows = sorted(
        (
            (
                str(name),
                int(geds[name].location.string),
                int(geds[name].location.position),
            )
            for name in geds
        ),
        key=lambda t: (t[1], t[2]),
    )
    return [n for n, _, _ in rows], [s for _, s, _ in rows]


def _grid(viewer, datatype):
    """Shared row/column scaffolding for both matrices."""
    cols = _columns(viewer, datatype)
    first_key = next((sk for _, _, sk, _ in cols if sk is not None), None)
    if first_key is None:
        msg = f"no runs of this cycle have a {datatype!r} entry in runinfo"
        raise KeyError(msg)
    names, strings = _detector_rows(viewer, first_key)
    return cols, names, strings


def _fmt_livetime(seconds):
    if seconds is None:
        return "—"
    return f"{seconds / 86400:.2f} d"


# ---------------------------------------------------------------------------
# usability matrix
# ---------------------------------------------------------------------------


def usability_cells(viewer, datatype="phy"):
    """Long-form per-cell columns for the usability matrix + grid scaffolding."""
    cols, names, strings = _grid(viewer, datatype)
    statuses_db = viewer.status_db.statuses

    cells = {c: [] for c in ("x", "y", "color", "usability", "reason", "livetime")}
    for period, run, start_key, livetime in cols:
        if start_key is None:
            continue  # run absent from runinfo for this datatype -> blank column
        statuses = statuses_db.on(start_key, category=datatype)
        for name in names:
            entry = statuses.get(name)
            if entry is None:
                continue  # not in the status DB -> blank cell
            usability = str(entry.get("usability", ""))
            cells["x"].append((period, run))
            cells["y"].append(name)
            cells["color"].append(STATUS_COLORS.get(usability, UNKNOWN_COLOR))
            cells["usability"].append(usability)
            cells["reason"].append(str(entry.get("reason", "")) or "—")
            cells["livetime"].append(_fmt_livetime(livetime))
    return cells, cols, names, strings


def usability_figure(viewer, datatype="phy"):
    """Detector x run usability heatmap; hover shows the off/ac reason."""
    cells, cols, names, strings = usability_cells(viewer, datatype)
    tooltips = [
        ("detector", "@y"),
        ("run", "@x"),
        ("usability", "@usability"),
        ("reason", "@reason"),
        ("livetime", "@livetime"),
    ]
    fig, source = _build_matrix(
        cells, cols, names, strings, f"detector usability ({datatype})", tooltips
    )
    _add_swatch_legend(fig, list(STATUS_COLORS.items()))
    return fig, source


# ---------------------------------------------------------------------------
# psd status matrix
# ---------------------------------------------------------------------------


def _bb_like_status(psd):
    """Overall PSD readiness from the ``is_bb_like`` expression.

    Evaluates each flag named in the expression (ported from
    ``legendmeta.vis.psd``): any ``missing`` -> "missing", else any
    ``present`` (uncalibrated) -> "present", else "valid". Returns
    ``(status|None, expr)``.
    """
    expr = str(psd.get("is_bb_like", "") or "")
    status_map = psd.get("status", {})
    if not expr or expr == "missing":
        return None, expr or "—"
    fields = [f.strip() for f in expr.split("&")]
    statuses = {str(status_map.get(f, "missing")) for f in fields}
    if "missing" in statuses:
        overall = "missing"
    elif "present" in statuses:
        overall = "present"
    else:
        overall = "valid"
    return overall, expr


def psd_cells(viewer, datatype="phy"):
    """Long-form per-cell columns for the PSD-status matrix + scaffolding."""
    cols, names, strings = _grid(viewer, datatype)
    statuses_db = viewer.status_db.statuses

    cells = {c: [] for c in ("x", "y", "color", "status", "expr", "reason")}
    for period, run, start_key, _livetime in cols:
        if start_key is None:
            continue
        statuses = statuses_db.on(start_key, category=datatype)
        for name in names:
            entry = statuses.get(name)
            psd = entry.get("psd") if entry is not None else None
            if psd is None:
                continue  # no psd block -> blank cell
            overall, expr = _bb_like_status(psd)
            if overall is None:
                continue  # is_bb_like undefined -> blank cell
            cells["x"].append((period, run))
            cells["y"].append(name)
            cells["color"].append(PSD_COLORS[overall])
            cells["status"].append(overall)
            cells["expr"].append(expr)
            cells["reason"].append(str(entry.get("reason", "")) or "—")
    return cells, cols, names, strings


def psd_figure(viewer, datatype="phy"):
    """Detector x run PSD-readiness heatmap (``is_bb_like`` evaluation)."""
    cells, cols, names, strings = psd_cells(viewer, datatype)
    tooltips = [
        ("detector", "@y"),
        ("run", "@x"),
        ("psd", "@status"),
        ("is_bb_like", "@expr"),
        ("reason", "@reason"),
    ]
    fig, source = _build_matrix(
        cells, cols, names, strings, f"PSD status ({datatype})", tooltips
    )
    _add_swatch_legend(fig, list(PSD_COLORS.items()))
    return fig, source


# ---------------------------------------------------------------------------
# partitions matrix
# ---------------------------------------------------------------------------


def partitions_cells(viewer, datatype="phy"):
    """Long-form per-cell columns for the partitions matrix + scaffolding."""
    cols, names, strings = _grid(viewer, datatype)
    groupings = viewer.status_db[f"{datatype}_groupings"]
    default = groupings.get("default", {})

    merged = {
        name: merge_with_defaults(groupings.get(name, {}), default) for name in names
    }
    # colour by the partitions actually visible in this cycle's runs (the
    # groupings cover the whole catalogue, most of it outside the grid)
    run_set = {(p, r) for p, r, _, _ in cols}
    partitions = sorted(
        {
            part
            for m in merged.values()
            for (period, run), part in m.items()
            if (period, run) in run_set
        },
        key=lambda p: (partition_num(p), p),
    )
    palette = Category20[20]
    color_of = {p: palette[k % len(palette)] for k, p in enumerate(partitions)}

    cells = {c: [] for c in ("x", "y", "color", "partition", "label", "span")}
    for name in names:
        spans: dict = {}
        for (period, run), part in sorted(merged[name].items()):
            spans.setdefault(part, []).append(f"{period} {run}")
        for period, run, _sk, _lt in cols:
            part = merged[name].get((period, run))
            if part is None:
                continue  # run not in any partition for this detector
            runs = spans[part]
            cells["x"].append((period, run))
            cells["y"].append(name)
            cells["color"].append(color_of[part])
            cells["partition"].append(part)
            cells["label"].append(partition_label(part))
            cells["span"].append(
                f"{runs[0]} … {runs[-1]}" if len(runs) > 1 else runs[0]
            )
    return cells, cols, names, strings, partitions, color_of


def partitions_figure(viewer, datatype="phy"):
    """Detector x run partition-membership heatmap."""
    cells, cols, names, strings, partitions, color_of = partitions_cells(
        viewer, datatype
    )
    tooltips = [
        ("detector", "@y"),
        ("run", "@x"),
        ("partition", "@partition (@label)"),
        ("span", "@span"),
    ]
    title = f"analysis partitions ({datatype})"
    if not cells["x"]:
        title += " — no partitions cover this cycle's runs"
    fig, source = _build_matrix(cells, cols, names, strings, title, tooltips)
    _add_swatch_legend(fig, [(partition_label(p), color_of[p]) for p in partitions])
    return fig, source


# ---------------------------------------------------------------------------
# shared figure assembly
# ---------------------------------------------------------------------------


def _build_matrix(cells, cols, names, strings, title, tooltips):
    factors = [(p, r) for p, r, _, _ in cols]
    fig = figure(
        x_range=FactorRange(*factors),
        y_range=list(reversed(names)),  # first string at the top
        height=90 + ROW_PX * len(names),
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=title,
        x_axis_location="above",
    )
    fig.grid.visible = False
    fig.xaxis.major_label_orientation = 1.0
    fig.yaxis.major_label_text_font_size = "8px"
    fig.xaxis.major_label_text_font_size = "9px"
    fig.xaxis.group_text_font_size = "10px"

    # horizontal separators between detector strings (synthetic categorical
    # coords: the row for names[i] is centred at len(names) - 1 - i + 0.5)
    for i in range(1, len(strings)):
        if strings[i] != strings[i - 1]:
            fig.add_layout(
                Span(
                    location=len(names) - i,
                    dimension="width",
                    line_color="#666666",
                    line_width=1.2,
                )
            )

    source = ColumnDataSource(cells)
    rects = fig.rect(
        x="x",
        y="y",
        width=0.94,
        height=0.86,
        source=source,
        fill_color="color",
        line_color=None,
        # tap-to-jump: highlight the picked cell, don't dim the rest
        nonselection_fill_alpha=1.0,
        selection_line_color="#000000",
        selection_line_width=2,
    )
    fig.add_tools(HoverTool(renderers=[rects], tooltips=tooltips))
    fig.add_tools(TapTool(renderers=[rects]))
    return fig, source


def _add_swatch_legend(fig, items):
    """A colour-swatch legend from ``[(label, color), ...]`` dummy glyphs."""
    if not items:  # e.g. no partition covers this cycle's runs
        return
    for label, color in items:
        fig.scatter(
            x=[], y=[], marker="square", size=8, color=color, legend_label=label
        )
    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "8px"
    fig.legend.glyph_height = 12
    fig.legend.glyph_width = 12
    fig.legend.spacing = 0
    fig.add_layout(fig.legend[0], "right")


# ---------------------------------------------------------------------------
# exposure plot
# ---------------------------------------------------------------------------


def exposure_cells(viewer, datatype="phy"):
    """Per-run exposure columns: active mass x livetime, plus the cumulative sum.

    Active mass counts detectors with usability "on" at the run start. Runs
    without a runinfo entry (or without livetime, e.g. cal) contribute zero.
    """
    cols, names, _strings = _grid(viewer, datatype)
    statuses_db = viewer.status_db.statuses

    cells = {c: [] for c in ("x", "exposure", "cumulative", "mass", "livetime", "n_on")}
    total = 0.0
    for period, run, start_key, livetime in cols:
        mass_kg = 0.0
        n_on = 0
        if start_key is not None:
            statuses = statuses_db.on(start_key, category=datatype)
            chmap = viewer._channelmap(start_key)
            geds = chmap.map("system", unique=False).geds.map("name")
            for name in names:
                entry = statuses.get(name)
                if entry is None or str(entry.get("usability")) != "on":
                    continue
                n_on += 1
                if name in geds:
                    mass_kg += float(geds[name].production.mass_in_g) / 1000
        exposure = mass_kg * (livetime or 0) / SECONDS_PER_YEAR  # kg yr
        total += exposure
        cells["x"].append((period, run))
        cells["exposure"].append(exposure)
        cells["cumulative"].append(total)
        cells["mass"].append(mass_kg)
        cells["livetime"].append(_fmt_livetime(livetime))
        cells["n_on"].append(n_on)
    return cells, cols


def exposure_figure(viewer, datatype="phy"):
    """Per-run exposure bars with a cumulative-exposure line (kg yr)."""
    cells, cols = exposure_cells(viewer, datatype)
    factors = [(p, r) for p, r, _, _ in cols]
    source = ColumnDataSource(cells)
    fig = figure(
        x_range=FactorRange(*factors),
        height=380,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"exposure ({datatype})",
        y_axis_label="run exposure (kg yr)",
    )
    fig.xaxis.major_label_orientation = 1.0
    fig.xaxis.major_label_text_font_size = "9px"
    fig.xaxis.group_text_font_size = "10px"
    fig.xgrid.visible = False

    bars = fig.vbar(
        x="x",
        top="exposure",
        width=0.85,
        source=source,
        fill_color=LEGEND_BLUE,
        line_color=LEGEND_BLUE,
        fill_alpha=0.85,
        nonselection_fill_alpha=0.85,
        selection_line_color="#d62728",
        selection_line_width=2,
        legend_label="per run",
    )

    # cumulative line on its own axis (different scale than per-run bars)
    cumulative = cells["cumulative"]
    fig.extra_y_ranges = {
        "cumulative": Range1d(0, (max(cumulative) if cumulative else 1) * 1.1 or 1)
    }
    fig.add_layout(
        LinearAxis(y_range_name="cumulative", axis_label="cumulative (kg yr)"),
        "right",
    )
    fig.line(
        x="x",
        y="cumulative",
        source=source,
        y_range_name="cumulative",
        color="#d62728",
        line_width=2,
        legend_label="cumulative",
    )
    fig.scatter(
        x="x",
        y="cumulative",
        source=source,
        y_range_name="cumulative",
        color="#d62728",
        size=5,
        legend_label="cumulative",
    )

    fig.legend.location = "top_left"
    fig.legend.label_text_font_size = "9px"
    fig.add_tools(
        HoverTool(
            renderers=[bars],
            tooltips=[
                ("run", "@x"),
                ("exposure", "@exposure{0.000} kg yr"),
                ("cumulative", "@cumulative{0.000} kg yr"),
                ("active mass", "@mass{0.0} kg (@n_on dets)"),
                ("livetime", "@livetime"),
            ],
        )
    )
    fig.add_tools(TapTool(renderers=[bars]))
    return fig, source


# ---------------------------------------------------------------------------
# data-blocks timeline
# ---------------------------------------------------------------------------

#: Visual width for a trailing block whose livetime is unknown.
_NO_LIVETIME_WIDTH = timedelta(hours=6)
#: Floor so blocks squeezed by an equal next-start stay visible/hoverable.
_MIN_BLOCK_WIDTH = timedelta(minutes=5)

#: Block colours per datatype; other datatypes get palette fallbacks.
DATATYPE_COLORS = {"phy": LEGEND_BLUE, "cal": "#FFA500"}


def timeline_cells(viewer):
    """One block per run *and datatype* on a real time axis.

    Unlike the matrices this view is not filtered by datatype: every runinfo
    entry of every run with data (phy, cal, ...) becomes a block, so the
    cal/phy alternation is visible in one plot. Data taking is sequential —
    datatypes never overlap in time — so a block without a livetime (cal,
    fft, ...) runs from its start to the next block's start (the last one
    gets a nominal width).
    """
    runinfo = viewer.status_db.runinfo
    runs = viewer.available_runs()

    entries = [
        (period, run, dtype, info)
        for period in sorted(runs)
        for run in sorted(runs[period])
        for dtype, info in runinfo.get(period, {}).get(run, {}).items()
        if info.get("start_key") is not None
    ]
    if not entries:
        msg = "no runs of this cycle have a runinfo entry"
        raise KeyError(msg)

    seen = {dtype for _, _, dtype, _ in entries}
    dt_order = [d for d in ("phy", "cal") if d in seen]
    dt_order += sorted(seen - set(dt_order))
    fallback = iter(Category10[10])
    color_of = {d: DATATYPE_COLORS.get(d) or next(fallback) for d in dt_order}

    starts = {
        (period, run, dtype): datetime.fromtimestamp(
            _tstamp_to_unix(info["start_key"]), UTC
        )
        for period, run, dtype, info in entries
    }
    entries.sort(key=lambda e: starts[e[:3]])  # chronological, across datatypes

    cells = {
        c: []
        for c in (
            "x",
            "y",
            "left",
            "right",
            "color",
            "datatype",
            "label",
            "start",
            "livetime",
        )
    }
    for k, (period, run, dtype, info) in enumerate(entries):
        livetime = info.get("livetime_in_s")
        start = starts[(period, run, dtype)]
        if livetime:
            end = start + timedelta(seconds=livetime)
        elif k + 1 < len(entries):
            end = starts[entries[k + 1][:3]]  # runs until the next block starts
        else:
            end = start + _NO_LIVETIME_WIDTH
        end = max(end, start + _MIN_BLOCK_WIDTH)
        cells["x"].append((period, run))  # click-to-jump hook (not drawn)
        cells["y"].append(period)
        cells["left"].append(start)
        cells["right"].append(end)
        cells["color"].append(color_of[dtype])
        cells["datatype"].append(dtype)
        cells["label"].append(f"{period} {run}")
        cells["start"].append(start.strftime("%Y-%m-%d %H:%M UTC"))
        cells["livetime"].append(_fmt_livetime(livetime))
    periods = sorted({p for p, _, _, _ in entries})
    return cells, periods, [(d, color_of[d]) for d in dt_order]


def timeline_figure(viewer, datatype="phy"):  # noqa: ARG001 (shows all datatypes)
    """Data blocks vs time, one lane per period; hover shows each run's livetime."""
    cells, periods, datatype_colors = timeline_cells(viewer)
    source = ColumnDataSource(cells)
    fig = figure(
        x_axis_type="datetime",
        y_range=list(reversed(periods)),  # earliest period on top
        height=120 + 30 * len(periods),
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title="data blocks (all datatypes)",
    )
    fig.ygrid.visible = False
    blocks = fig.hbar(
        y="y",
        left="left",
        right="right",
        height=0.6,
        source=source,
        fill_color="color",
        line_color="color",
        fill_alpha=0.9,
        nonselection_fill_alpha=0.9,
        selection_line_color="#000000",
        selection_line_width=2,
    )
    fig.add_tools(
        HoverTool(
            renderers=[blocks],
            tooltips=[
                ("run", "@label"),
                ("datatype", "@datatype"),
                ("start", "@start"),
                ("livetime", "@livetime"),
            ],
        )
    )
    fig.add_tools(TapTool(renderers=[blocks]))
    _add_swatch_legend(fig, datatype_colors)
    return fig, source


_BUILDERS = {
    "data": timeline_figure,
    "exposure": exposure_figure,
    "usabilities": usability_figure,
    "partitions": partitions_figure,
    "psd": psd_figure,
}


def dataset_figure(viewer, plot=PLOTS[0], datatype="phy"):
    """Entry point: build the requested dataset-metadata figure.

    Returns ``(figure, cells_source)``; tapping a glyph selects an index in
    ``cells_source``, whose ``x`` column holds the ``(period, run)`` factor —
    the hook for click-to-jump navigation.
    """
    return _BUILDERS[plot](viewer, datatype)
