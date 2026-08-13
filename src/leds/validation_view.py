"""Bokeh views for the Validation tab: rate-vs-time series and calibration.

Rate figures share one builder (datetime x-axis, one line per series, NaN
gaps between runs, click-to-hide legend). The K-line view stacks a K40 and a
K42 figure with a linked x-range. The calibration views draw residuals vs
detector (summary) and one detector's ADC-to-keV curve (detail) from the data
prepared in :mod:`leds.validation`.

Framework-agnostic: builds Bokeh figures, no Panel.
"""

from __future__ import annotations

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure

from leds.validation import (
    GROUP_UNIT_LABEL,
    GROUP_UNIT_SECONDS,
    K_LINES,
    RATE_GROUPS,
    survival_fraction,
)

PLOTS = (
    "trigger rates",
    "multiplicity rates",
    "K-line rates",
    "qc survival",
    "calibration summary",
    "calibration detail",
)

LEGEND_BLUE = "#1A2A5B"
_PALETTE = Category10[10]

#: Peak colours in the calibration summary (matched to PEAKS_SUMMARY order).
_PEAK_COLORS = (LEGEND_BLUE, "#6BAE75", "#FF4444")


def _rates_figure(times_ms, rates, group, *, title, height=340, log_y=True):
    """One rate-vs-time figure for every available series of ``group``.

    ``log_y`` (user-toggleable) suits the decades the rates span (triggers vs
    muons, K lines across cuts); zero-rate bins have no log-scale position
    and are simply not drawn there.
    """
    fig = figure(
        x_axis_type="datetime",
        y_axis_type="log" if log_y else "linear",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=title,
    )
    fig.yaxis.axis_label = GROUP_UNIT_LABEL[GROUP_UNIT_SECONDS[group]]
    missing = []
    for i, label in enumerate(RATE_GROUPS[group]):
        series = rates.get((group, label))
        if series is None:
            missing.append(label)
            continue
        source = ColumnDataSource({"t": times_ms, "rate": series})
        color = _PALETTE[i % len(_PALETTE)]
        fig.line("t", "rate", source=source, color=color, legend_label=label)
        pts = fig.scatter(
            "t", "rate", source=source, color=color, size=4, legend_label=label
        )
        fig.add_tools(
            HoverTool(
                renderers=[pts],
                tooltips=[
                    ("series", label),
                    ("time", "@t{%F %T}"),
                    ("rate", "@rate{0.000}"),
                ],
                formatters={"@t": "datetime"},
            )
        )
    if missing:
        fig.title.text += f"  (unavailable: {', '.join(missing)})"
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_right"
    return fig


def trigger_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    return _rates_figure(
        times_ms,
        rates,
        "trigger",
        title=f"trigger rates ({bin_label} bins, {scope})",
        log_y=log_y,
    )


def multiplicity_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    return _rates_figure(
        times_ms,
        rates,
        "multiplicity",
        title=f"multiplicity rates, forced/pulser removed "
        f"({bin_label} bins, {scope})",
        log_y=log_y,
    )


def qc_survival(times_ms, rates, bin_label, log_y=False, scope="all strings"):
    """Fraction of physics events passing the quality cuts, per time bin.

    A ratio of two same-scope rates, so the mass normalisation cancels; the
    string restriction still applies through the underlying series.
    """
    frac = survival_fraction(rates.get(("qc", "pass")), rates.get(("qc", "fail")))
    fig = figure(
        x_axis_type="datetime",
        y_axis_type="log" if log_y else "linear",
        height=340,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"quality-cut survival fraction, forced/pulser removed "
        f"({bin_label} bins, {scope})",
        # a fixed 0..1 range only makes sense on a linear axis
        **({} if log_y else {"y_range": (0.0, 1.05)}),
    )
    fig.yaxis.axis_label = "survival fraction"
    if frac is None:
        fig.title.text += "  (quality flags unavailable)"
        return fig
    source = ColumnDataSource({"t": times_ms, "frac": frac})
    fig.line("t", "frac", source=source, color=LEGEND_BLUE)
    pts = fig.scatter("t", "frac", source=source, color=LEGEND_BLUE, size=4)
    fig.add_tools(
        HoverTool(
            renderers=[pts],
            tooltips=[("time", "@t{%F %T}"), ("survival", "@frac{0.0000}")],
            formatters={"@t": "datetime"},
        )
    )
    return fig


def kline_rates(times_ms, rates, bin_label, log_y=True, scope="all strings"):
    """K40 and K42 line rates, stacked with a linked x-range."""
    figs = []
    for line, peak in K_LINES.items():
        fig = _rates_figure(
            times_ms,
            rates,
            line,
            title=f"{line} line rate, {peak:.1f} keV, forced/pulser removed "
            f"({bin_label} bins, {scope})",
            height=280,
            log_y=log_y,
        )
        if figs:
            fig.x_range = figs[0].x_range
        figs.append(fig)
    return column(*figs, sizing_mode="stretch_width")


#: Rate-plot label -> builder(times_ms, rates, bin_label).
RATE_BUILDERS = {
    "trigger rates": trigger_rates,
    "multiplicity rates": multiplicity_rates,
    "K-line rates": kline_rates,
    "qc survival": qc_survival,
}


def cal_summary(names, residuals, strings, source_label):
    """Calibration residuals vs detector, one series per summary peak.

    ``strings`` maps each detector index to its string number (for boundary
    separators); pass ``None`` to skip them.
    """
    x = np.arange(len(names))
    fig = figure(
        height=380,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"calibration residuals per detector ({source_label})",
        x_range=(-1.0, len(names) + 0.5),
    )
    fig.yaxis.axis_label = "residual (keV)"
    fig.xaxis.ticker = x
    fig.xaxis.major_label_overrides = {int(i): n for i, n in zip(x, names)}
    fig.xaxis.major_label_orientation = 1.2
    if strings is not None:
        for i in range(1, len(strings)):
            if strings[i] != strings[i - 1]:
                fig.add_layout(
                    Span(
                        location=i - 0.5,
                        dimension="height",
                        line_color="#BBBBBB",
                        line_dash="dashed",
                    )
                )
    for color, (peak, (res, err)) in zip(_PEAK_COLORS, residuals.items()):
        source = ColumnDataSource(
            {
                "x": x,
                "res": res,
                "name": names,
                "err_xs": [[i, i] for i in x],
                "err_ys": [[r - e, r + e] for r, e in zip(res, err)],
            }
        )
        label = f"{peak:.1f} keV"
        pts = fig.scatter(
            "x", "res", source=source, color=color, size=6, legend_label=label
        )
        fig.multi_line("err_xs", "err_ys", source=source, color=color, alpha=0.6)
        fig.add_tools(
            HoverTool(
                renderers=[pts],
                tooltips=[
                    ("detector", "@name"),
                    ("peak", label),
                    ("residual", "@res{0.000} keV"),
                ],
            )
        )
    fig.legend.click_policy = "hide"
    return fig


def cal_detail(curve, detector, source_label):
    """One detector's calibration curve (top) and residuals (bottom)."""
    top = figure(
        height=320,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title=f"{detector} energy calibration ({source_label}): "
        f"{curve['expression']}",
    )
    top.xaxis.axis_label = "peak centroid (ADC)"
    top.yaxis.axis_label = "peak energy (keV)"
    top.line(curve["line_x"], curve["line_y"], color="#888888", line_dash="dashed")
    source = ColumnDataSource(
        {
            "mu": curve["mu"],
            "peak": curve["peaks"],
            "res": curve["residual"],
            "err": curve["cal_err"],
            "res_xs": [[p, p] for p in curve["peaks"]],
            "res_ys": [
                [r - e, r + e] for r, e in zip(curve["residual"], curve["cal_err"])
            ],
        }
    )
    pts = top.scatter("mu", "peak", source=source, color=LEGEND_BLUE, size=8)
    top.add_tools(
        HoverTool(
            renderers=[pts],
            tooltips=[("peak", "@peak{0.000} keV"), ("centroid", "@mu{0.0} ADC")],
        )
    )

    bottom = figure(
        height=240,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="right",
        title="residuals",
    )
    bottom.xaxis.axis_label = "peak energy (keV)"
    bottom.yaxis.axis_label = "calibrated - true (keV)"
    bottom.add_layout(Span(location=0, dimension="width", line_color="#BBBBBB"))
    res_pts = bottom.scatter("peak", "res", source=source, color=LEGEND_BLUE, size=8)
    bottom.multi_line("res_xs", "res_ys", source=source, color=LEGEND_BLUE, alpha=0.6)
    bottom.add_tools(
        HoverTool(
            renderers=[res_pts],
            tooltips=[
                ("peak", "@peak{0.000} keV"),
                ("residual", "@res{0.000} keV"),
                ("error", "@err{0.000} keV"),
            ],
        )
    )
    return column(top, bottom, sizing_mode="stretch_width")
