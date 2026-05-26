"""
ISAPlotter — six standard diagnostic / lifecycle plots for ISA-PHM data.

All methods return a Bokeh figure or layout object.  Call::

    from bokeh.io import output_notebook
    from bokeh.plotting import show
    output_notebook()   # once, at the top of the notebook
    show(fig)           # to display each figure

They never call ``show()`` directly — the caller controls display or saving.

The six MVP plots (SR-4)
------------------------
1. plot_distribution     — amplitude histogram + KDE for one run's time series.
2. plot_lifecycle        — scalar feature across all runs (lifecycle curve).
3. plot_frequency_domain — FFT magnitude spectrum for one run.
4. plot_correlation      — correlation heatmap of lifecycle scalar features.
5. plot_variability      — boxplot of amplitude by run_id.
6. plot_missing_values   — heatmap of NaN presence across DataFrame columns.

Additional
----------
plot_timeseries          — time-domain waveform with optional outlier overlay.
plot_outlier_comparison  — linked before/after waveform panels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import fft as sp_fft
from scipy.stats import gaussian_kde

from .errors import PlotError
from .utils import FEATURE_NAMES

try:
    from bokeh.layouts import column as bk_column, gridplot
    from bokeh.models import (
        BasicTicker,
        ColorBar,
        ColumnDataSource,
        Div,
        HoverTool,
        LinearColorMapper,
        Whisker,
    )
    from bokeh.plotting import figure as bokeh_figure
    import bokeh.palettes as _bokeh_palettes
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "bokeh is required for ISAPlotter. "
        "Install it with: pip install bokeh"
    ) from _exc

logger = logging.getLogger("isa_phm")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Global style / layout settings shared by all ISAPlotter methods."""

    width: int = 1150
    height: int = 400
    title_fontsize: str = "13pt"
    label_fontsize: str = "11pt"
    line_color: str = "#4C72B0"
    outlier_color: str = "firebrick"


# ---------------------------------------------------------------------------
# ISAPlotter
# ---------------------------------------------------------------------------

class ISAPlotter:
    """
    Stateless (except for config) plotting helper for ISA-PHM DataFrames.

    Parameters
    ----------
    config : PlotConfig | None
        If None, default PlotConfig() is used.
    """

    def __init__(self, config: PlotConfig | None = None) -> None:
        self._cfg = config or PlotConfig()

    # ------------------------------------------------------------------
    # 1. plot_distribution
    # ------------------------------------------------------------------

    def plot_distribution(
        self,
        df: pd.DataFrame,
        column: str = "value",
        bins: int = 50,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Histogram + KDE of ``column`` in ``df``.

        Parameters
        ----------
        df : pd.DataFrame
        column : str
        bins : int
        title : str | None

        Returns
        -------
        bokeh.plotting.figure
        """
        self._require_column(df, column, "plot_distribution")
        values = df[column].dropna().to_numpy(dtype=float)
        self._require_nonempty(values, column, "plot_distribution")

        hist, edges = np.histogram(values, bins=bins, density=True)

        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Distribution of '{column}'",
            x_axis_label=xlabel or column,
            y_axis_label=ylabel or "Density",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize

        p.quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color=self._cfg.line_color, line_color="white",
            alpha=0.7, legend_label="Histogram",
        )

        if len(np.unique(values)) > 5:
            try:
                kde = gaussian_kde(values)
                x_kde = np.linspace(values.min(), values.max(), 500)
                y_kde = kde(x_kde)
                p.line(x_kde, y_kde, color="firebrick", line_width=2, legend_label="KDE")
            except (np.linalg.LinAlgError, ValueError):
                pass  # KDE can fail on degenerate data; histogram is sufficient.

        p.add_tools(HoverTool(tooltips=[((ylabel or "Density"), "@top{0.000000}")]))
        p.legend.location = "top_right"
        return p

    # ------------------------------------------------------------------
    # 2. plot_lifecycle
    # ------------------------------------------------------------------

    def plot_lifecycle(
        self,
        lifecycle_df: pd.DataFrame,
        feature: str = "rms",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Line plot of a scalar feature across runs (lifecycle curve).

        Parameters
        ----------
        lifecycle_df : pd.DataFrame
            Output of ``DataIntegrator.lifecycle_features_df()``.
        feature : str
        title : str | None

        Returns
        -------
        bokeh.plotting.figure
        """
        for col in ("run_number", feature):
            self._require_column(lifecycle_df, col, "plot_lifecycle")

        df_sorted = lifecycle_df.sort_values("run_number")
        source = ColumnDataSource(dict(
            x=df_sorted["run_number"].tolist(),
            y=df_sorted[feature].tolist(),
        ))

        y_label = ylabel or feature.upper()
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Lifecycle — {feature.upper()}",
            x_axis_label=xlabel or "Run number",
            y_axis_label=y_label,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.line("x", "y", source=source, color=self._cfg.line_color, line_width=1.8)
        p.scatter("x", "y", source=source, color=self._cfg.line_color, size=6)
        p.add_tools(HoverTool(tooltips=[
            ("Run", "@x"),
            (y_label, "@y{0.000000}"),
        ]))
        return p

    def plot_multi_lifecycle(
        self,
        lifecycle_dfs: "dict[str, pd.DataFrame]",
        feature: str = "rms",
        title: str | None = None,
        palette: "list[str] | None" = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Overlay lifecycle curves for multiple assays or bearings on one figure.

        Parameters
        ----------
        lifecycle_dfs : dict[str, pd.DataFrame]
            Mapping of ``{label: lifecycle_df}``.  Each DataFrame must have
            columns ``run_number`` and ``feature``.
        feature : str
            Scalar feature column to plot (default ``"rms"``).
        title : str | None
        palette : list[str] | None
            Line/marker colours.  Defaults to Bokeh Category10.

        Returns
        -------
        bokeh.plotting.figure
        """
        if not lifecycle_dfs:
            raise PlotError("lifecycle_dfs is empty — nothing to plot.")

        n = len(lifecycle_dfs)
        colors = palette or list(_bokeh_palettes.Category10[max(3, min(10, n))])
        if len(colors) < n:
            colors = (colors * (n // len(colors) + 1))[:n]

        y_label = ylabel or feature.upper()
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Prognostic Lifecycle Comparison — {feature.upper()}",
            x_axis_label=xlabel or "Run number",
            y_axis_label=y_label,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize

        for (label, lc_df), color in zip(lifecycle_dfs.items(), colors):
            if lc_df.empty or feature not in lc_df.columns:
                logger.warning("plot_multi_lifecycle: skipping '%s': missing '%s'.", label, feature)
                continue
            df_sorted = lc_df.sort_values("run_number")
            src = ColumnDataSource(dict(
                x=df_sorted["run_number"].tolist(),
                y=df_sorted[feature].tolist(),
                label=[label] * len(df_sorted),
            ))
            p.line("x", "y", source=src, color=color, line_width=1.8, legend_label=label)
            p.scatter("x", "y", source=src, color=color, size=5)

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.add_tools(HoverTool(tooltips=[
            ("Run",    "@x"),
            (y_label, "@y{0.000000}"),
            ("Label",  "@label"),
        ]))
        return p

    # ------------------------------------------------------------------
    # 3. plot_frequency_domain
    # ------------------------------------------------------------------

    def plot_frequency_domain(
        self,
        df: pd.DataFrame,
        fs: float,
        column: str = "value",
        title: str | None = None,
        log_scale: bool = True,
        unit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Single-sided FFT amplitude spectrum.

        Parameters
        ----------
        df : pd.DataFrame
        fs : float
            Sampling frequency in Hz.
        column : str
        title : str | None
        log_scale : bool
            True → magnitude in dB.  False → linear amplitude.

        Returns
        -------
        bokeh.plotting.figure
        """
        if fs is None or fs <= 0:
            raise PlotError(
                f"fs (sampling frequency) must be a positive float. Got: {fs!r}. "
                f"Pass fs= explicitly or ensure the ISA-JSON protocol parameters "
                f"include sampling frequency in Hz."
            )

        self._require_column(df, column, "plot_frequency_domain")
        values = df[column].dropna().to_numpy(dtype=float)
        self._require_nonempty(values, column, "plot_frequency_domain")

        n = len(values)
        yf = sp_fft.rfft(values - values.mean())
        magnitude = np.abs(yf) / n
        freqs = sp_fft.rfftfreq(n, d=1.0 / fs)

        if log_scale:
            y = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
            default_ylabel = (
                f"Spectral magnitude (dB re 1 {unit})"
                if unit
                else "Spectral magnitude (dB)"
            )
        else:
            y = magnitude
            default_ylabel = f"Spectral amplitude ({unit})" if unit else "Spectral amplitude"
        y_label = ylabel or default_ylabel

        source = ColumnDataSource(dict(x=freqs.tolist(), y=y.tolist()))
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"FFT Spectrum — '{column}'",
            x_axis_label=xlabel or "Frequency (Hz)",
            y_axis_label=y_label,
            x_range=(0, fs / 2),
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.line("x", "y", source=source, color=self._cfg.line_color, line_width=0.9)
        p.add_tools(HoverTool(tooltips=[
            ("Freq (Hz)", "@x{0.0}"),
            (y_label, "@y{0.000000}"),
        ]))
        return p

    # ------------------------------------------------------------------
    # 3b. plot_power_spectral_density
    # ------------------------------------------------------------------

    def plot_power_spectral_density(
        self,
        df: pd.DataFrame,
        fs: float,
        column: str = "value",
        nperseg: int = 1024,
        window: str = "hann",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Welch Power Spectral Density plot.

        Parameters
        ----------
        df : pd.DataFrame
        fs : float
            Sampling frequency in Hz.
        column : str
        nperseg : int
            Length of each Welch segment (default 1024).
        window : str
            Window function name passed to ``scipy.signal.welch`` (default "hann").

        Returns
        -------
        bokeh.plotting.figure
        """
        from scipy.signal import welch as scipy_welch

        if fs is None or fs <= 0:
            raise PlotError(
                f"fs (sampling frequency) must be a positive float. Got: {fs!r}. "
                f"Pass fs= explicitly or ensure the ISA-JSON protocol parameters "
                f"include sampling frequency in Hz."
            )
        self._require_column(df, column, "plot_power_spectral_density")
        values = df[column].dropna().to_numpy(dtype=float)
        self._require_nonempty(values, column, "plot_power_spectral_density")

        freqs, psd = scipy_welch(
            values,
            fs=fs,
            window=window,
            nperseg=min(nperseg, len(values)),
        )
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

        source = ColumnDataSource(dict(x=freqs.tolist(), y=psd_db.tolist()))
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Welch PSD — '{column}'",
            x_axis_label=xlabel or "Frequency (Hz)",
            y_axis_label=ylabel or "Power Spectral Density (dB/Hz)",
            x_range=(0, fs / 2),
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.line("x", "y", source=source, color=self._cfg.line_color, line_width=0.9)
        p.add_tools(HoverTool(tooltips=[
            ("Freq (Hz)", "@x{0.0}"),
            ("PSD (dB/Hz)", "@y{0.000}"),
        ]))
        return p

    # ------------------------------------------------------------------
    # 3c. plot_spectrogram
    # ------------------------------------------------------------------

    def plot_spectrogram(
        self,
        df: pd.DataFrame,
        fs: float,
        column: str = "value",
        nperseg: int = 256,
        overlap: float = 0.75,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Short-Time Fourier Transform spectrogram (time × frequency heatmap).

        Parameters
        ----------
        df : pd.DataFrame
        fs : float
            Sampling frequency in Hz.
        column : str
            Signal column name.
        nperseg : int
            FFT window length in samples.
        overlap : float
            Fraction of window overlap (0–1).  Default 0.75.
        """
        from scipy.signal import spectrogram as scipy_spectrogram
        import numpy as np
        from bokeh.models import LinearColorMapper, ColorBar
        from bokeh.palettes import Viridis256

        self._require_column(df, column, "plot_spectrogram")
        values = df[column].dropna().to_numpy(dtype=float)
        self._require_nonempty(values, column, "plot_spectrogram")

        noverlap = int(nperseg * overlap)
        freqs, times, Sxx = scipy_spectrogram(
            values, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-30))

        mapper = LinearColorMapper(
            palette=Viridis256,
            low=float(Sxx_db.min()),
            high=float(Sxx_db.max()),
        )
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Spectrogram — '{column}'",
            x_axis_label="Time (s)",
            y_axis_label="Frequency (Hz)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.image(
            image=[Sxx_db],
            x=float(times[0]),
            y=float(freqs[0]),
            dw=float(times[-1] - times[0]),
            dh=float(freqs[-1] - freqs[0]),
            color_mapper=mapper,
        )
        color_bar = ColorBar(color_mapper=mapper, width=8)
        p.add_layout(color_bar, "right")
        return p

    # ------------------------------------------------------------------
    # 3d. plot_waterfall
    # ------------------------------------------------------------------

    def plot_waterfall(
        self,
        dfs: dict[str, pd.DataFrame],
        fs: float,
        column: str = "value",
        nperseg: int = 1024,
        offset_scale: float = 0.3,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Waterfall plot — stacked FFT spectra, one per run, offset vertically.

        Shows how the frequency content evolves across the lifecycle.

        Parameters
        ----------
        dfs : dict[str, pd.DataFrame]
            Mapping of run label → DataFrame.  Pass a sparse selection
            (e.g. every 10th run) for readability.
        fs : float
            Sampling frequency in Hz.
        column : str
            Signal column name.
        nperseg : int
            FFT window length (passed to Welch for smoothing).
        offset_scale : float
            Vertical offset between traces as a fraction of the max PSD range.
        """
        import numpy as np
        from scipy.signal import welch as scipy_welch
        from bokeh.palettes import Viridis
        from bokeh.models import HoverTool

        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or "Waterfall — FFT evolution",
            x_axis_label="Frequency (Hz)",
            y_axis_label="Power (dB/Hz, offset per run)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize

        n = len(dfs)
        palette = (Viridis[max(n, 3)] if n <= 256 else Viridis[256])[:n]

        psd_ranges = []
        spectra = []
        for label, df in dfs.items():
            if column not in df.columns:
                continue
            vals = df[column].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            freqs, psd = scipy_welch(vals, fs=fs, nperseg=min(nperseg, len(vals)))
            psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))
            psd_ranges.append(psd_db.max() - psd_db.min())
            spectra.append((label, freqs, psd_db))

        if not spectra:
            raise PlotError("plot_waterfall: no valid data found in provided DataFrames.")

        step = (sum(psd_ranges) / len(psd_ranges)) * offset_scale

        for i, (label, freqs, psd_db) in enumerate(spectra):
            offset = i * step
            color = palette[i % len(palette)]
            source = ColumnDataSource(dict(
                x=freqs.tolist(),
                y=(psd_db + offset).tolist(),
                label=[label] * len(freqs),
            ))
            p.line("x", "y", source=source, color=color, line_width=1.0,
                   alpha=0.85, legend_label=label if n <= 12 else None)

        if n <= 12:
            p.legend.location = "top_right"
            p.legend.label_text_font_size = "9pt"
        p.add_tools(HoverTool(tooltips=[("Run", "@label"), ("Freq (Hz)", "@x{0.0}"), ("PSD (dB/Hz)", "@y{0.00}")]))
        return p

    # ------------------------------------------------------------------
    # 4. plot_correlation
    # ------------------------------------------------------------------

    def plot_correlation(
        self,
        df: pd.DataFrame,
        columns: Sequence[str] | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Correlation heatmap of scalar features.

        Returns
        -------
        bokeh.plotting.figure
        """
        if columns is None:
            columns = [c for c in FEATURE_NAMES if c in df.columns]
        else:
            columns = [c for c in columns if c in df.columns]

        if len(columns) < 2:
            raise PlotError(
                f"plot_correlation needs at least 2 numeric columns. "
                f"Found in DataFrame: {list(df.columns)}."
            )
        sub = df[list(columns)].dropna()
        if len(sub) < 2:
            raise PlotError(
                f"plot_correlation needs at least 2 non-null rows. Got {len(sub)}."
            )

        corr = sub.corr()

        xs, ys, vals, texts = [], [], [], []
        for col_y in columns:
            for col_x in columns:
                xs.append(col_x)
                ys.append(col_y)
                val = float(corr.loc[col_y, col_x])
                vals.append(val)
                texts.append(f"{val:.2f}")

        source = ColumnDataSource(dict(x=xs, y=ys, values=vals, text=texts))
        mapper = LinearColorMapper(palette=_bokeh_palettes.RdBu11, low=-1, high=1)

        sz = max(300, 60 * len(columns))
        width_px = width or sz
        height_px = height or sz
        p = bokeh_figure(
            width=width_px,
            height=height_px,
            title=title or "Feature Correlation Matrix",
            x_range=list(columns),
            y_range=list(reversed(columns)),
            x_axis_label=xlabel,
            y_axis_label=ylabel,
            tools="save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.rect(
            x="x", y="y", width=1, height=1, source=source,
            fill_color={"field": "values", "transform": mapper},
            line_color=None,
        )
        p.text(
            x="x", y="y", text="text", source=source,
            text_align="center", text_baseline="middle",
            text_font_size="9pt",
        )
        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=5),
            label_standoff=6,
            major_label_text_font_size="9pt",
        )
        p.add_layout(color_bar, "right")
        p.xaxis.major_label_orientation = 0.785  # 45°
        return p

    # ------------------------------------------------------------------
    # 5. plot_variability
    # ------------------------------------------------------------------

    def plot_variability(
        self,
        df: pd.DataFrame,
        value_column: str = "value",
        group_by: str = "run_id",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Box plot of ``value_column`` grouped by ``group_by``.

        Returns
        -------
        bokeh.plotting.figure
        """
        for col in (value_column, group_by):
            self._require_column(df, col, "plot_variability")

        group_labels = [str(k) for k in df[group_by].unique()]
        q1s, medians, q3s, uppers, lowers, means = [], [], [], [], [], []
        for lbl in group_labels:
            vals = df.loc[df[group_by] == lbl, value_column].dropna()
            q1  = float(vals.quantile(0.25))
            med = float(vals.quantile(0.50))
            q3  = float(vals.quantile(0.75))
            iqr = q3 - q1
            q1s.append(q1)
            medians.append(med)
            q3s.append(q3)
            means.append(float(vals.mean()))
            uppers.append(min(float(vals.max()), q3 + 1.5 * iqr))
            lowers.append(max(float(vals.min()), q1 - 1.5 * iqr))

        if not group_labels:
            raise PlotError("plot_variability: no groups found. Check group_by column values.")

        source = ColumnDataSource(dict(
            x=group_labels, q1=q1s, median=medians, q3=q3s,
            mean=means, upper=uppers, lower=lowers,
        ))

        w = max(width or self._cfg.width, len(group_labels) * 60)
        p = bokeh_figure(
            x_range=group_labels,
            width=w,
            height=height or self._cfg.height,
            title=title or f"Variability of '{value_column}' by '{group_by}'",
            x_axis_label=xlabel or group_by,
            y_axis_label=ylabel or value_column,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.add_layout(Whisker(source=source, base="x", upper="upper", lower="lower",
                             line_color="#555555"))
        p.vbar(x="x", top="median", bottom="q1", width=0.6, source=source,
               fill_color="#4C72B0", line_color="black", fill_alpha=0.9,
               legend_label="Q1 – median")
        p.vbar(x="x", top="q3", bottom="median", width=0.6, source=source,
               fill_color="#6B9FD4", line_color="black", fill_alpha=0.9,
               legend_label="median – Q3")
        p.scatter(x="x", y="mean", size=8, source=source, color="orangered",
                  legend_label="Mean")
        p.add_tools(HoverTool(tooltips=[
            ("Group",   "@x"),
            ("Median",  "@median{0.00000}"),
            ("Mean",    "@mean{0.00000}"),
            ("Q1",      "@q1{0.00000}"),
            ("Q3",      "@q3{0.00000}"),
        ]))
        p.legend.location = "top_right"
        return p

    # ------------------------------------------------------------------
    # 6. plot_missing_values
    # ------------------------------------------------------------------

    def plot_missing_values(
        self,
        df: pd.DataFrame,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Heatmap of NaN presence across DataFrame columns.

        Returns
        -------
        bokeh.plotting.figure
        """
        if df.empty:
            raise PlotError("plot_missing_values: DataFrame is empty.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise PlotError("plot_missing_values: no numeric columns found.")

        sub = df[numeric_cols]
        missing_matrix = sub.isnull().astype(int)
        MAX_ROWS = 200
        stride = max(1, len(missing_matrix) // MAX_ROWS)
        sampled = missing_matrix.iloc[::stride]
        n_rows = len(sampled)

        xs, ys, vals = [], [], []
        for ci, col in enumerate(numeric_cols):
            for ri in range(n_rows):
                xs.append(col)
                ys.append(ri)
                vals.append(int(sampled.iloc[ri, ci]))

        source = ColumnDataSource(dict(x=xs, y=ys, v=vals))
        mapper = LinearColorMapper(palette=["#eaeaea", "#d73027"], low=0, high=1)

        p = bokeh_figure(
            x_range=numeric_cols,
            width=max(width or self._cfg.width, len(numeric_cols) * 80),
            height=height or max(300, min(600, n_rows * 3)),
            title=title or "Missing Values Heatmap",
            x_axis_label=xlabel or "Column",
            y_axis_label=ylabel or f"Row index (stride={stride})",
            tools="pan,wheel_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.rect(x="x", y="y", width=1, height=1, source=source,
               fill_color={"field": "v", "transform": mapper}, line_color=None)
        p.add_tools(HoverTool(tooltips=[
            ("Column",  "@x"),
            ("Row",     "@y"),
            ("Missing", "@v"),
        ]))
        return p

    # ------------------------------------------------------------------
    # plot_timeseries
    # ------------------------------------------------------------------

    def plot_timeseries(
        self,
        df: pd.DataFrame,
        time_column: str = "time",
        value_column: str = "value",
        outlier_mask: "np.ndarray | None" = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        max_points: int = 10_000,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Time-domain waveform plot with optional outlier overlay.

        Parameters
        ----------
        df : pd.DataFrame
        time_column, value_column : str
        outlier_mask : np.ndarray | None
            Boolean array aligned with df rows. Outlier samples are shown
            as red scatter points over the waveform.
        title, xlabel, ylabel : str | None
        max_points : int
            Down-sample to this many points for responsive rendering.

        Returns
        -------
        bokeh.plotting.figure
        """
        self._require_column(df, time_column, "plot_timeseries")
        self._require_column(df, value_column, "plot_timeseries")

        t = df[time_column].to_numpy(dtype=float)
        v = df[value_column].to_numpy(dtype=float)
        self._require_nonempty(v[np.isfinite(v)], value_column, "plot_timeseries")

        stride = max(1, len(t) // max_points)
        t_plot, v_plot = t[::stride], v[::stride]

        source = ColumnDataSource(dict(t=t_plot.tolist(), v=v_plot.tolist()))
        p = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=title or f"Time-domain waveform — '{value_column}'",
            x_axis_label=xlabel or time_column,
            y_axis_label=ylabel or value_column,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.title.text_font_size = self._cfg.title_fontsize
        p.line("t", "v", source=source, color=self._cfg.line_color, line_width=0.8,
               legend_label="Signal")

        if outlier_mask is not None and outlier_mask.any():
            mask_ds = outlier_mask[::stride]
            n_outliers_total = int(outlier_mask.sum())
            out_src = ColumnDataSource(dict(
                t=t_plot[mask_ds].tolist(),
                v=v_plot[mask_ds].tolist(),
            ))
            p.scatter("t", "v", source=out_src, color=self._cfg.outlier_color,
                      size=5, legend_label=f"Outliers ({n_outliers_total})")

        p.add_tools(HoverTool(tooltips=[
            ("Time", "@t{0.000}"),
            (ylabel or value_column, "@v{0.000000}"),
        ]))
        p.legend.location = "top_right"
        return p

    # ------------------------------------------------------------------
    # plot_outlier_comparison
    # ------------------------------------------------------------------

    def plot_outlier_comparison(
        self,
        df_original: pd.DataFrame,
        df_clean: pd.DataFrame,
        time_column: str = "time",
        value_column: str = "value",
        strategy: str = "correction",
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Linked before/after waveform panels (x-axes are synchronised).

        Returns
        -------
        bokeh layout (gridplot or column)
        """
        self._require_column(df_original, time_column, "plot_outlier_comparison")
        self._require_column(df_original, value_column, "plot_outlier_comparison")

        t_orig = df_original[time_column].to_numpy(dtype=float)
        v_orig = df_original[value_column].to_numpy(dtype=float)
        t_clean = (
            df_clean[time_column].to_numpy(dtype=float)
            if time_column in df_clean.columns
            else t_orig[: len(df_clean)]
        )
        v_clean = df_clean[value_column].to_numpy(dtype=float)

        s_orig  = max(1, len(t_orig)  // 10_000)
        s_clean = max(1, len(t_clean) // 10_000)

        xl = xlabel or time_column
        yl = ylabel or value_column

        src_orig  = ColumnDataSource(dict(t=t_orig[::s_orig].tolist(),   v=v_orig[::s_orig].tolist()))
        src_clean = ColumnDataSource(dict(t=t_clean[::s_clean].tolist(), v=v_clean[::s_clean].tolist()))

        p1 = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title="Original",
            x_axis_label=xl, y_axis_label=yl,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p1.line("t", "v", source=src_orig, color=self._cfg.line_color, line_width=0.8)
        p1.add_tools(HoverTool(tooltips=[("Time", "@t{0.000}"), (yl, "@v{0.000000}")]))

        p2 = bokeh_figure(
            width=width or self._cfg.width,
            height=height or self._cfg.height,
            title=f"After {strategy}",
            x_axis_label=xl, y_axis_label=yl,
            x_range=p1.x_range,   # linked x-axis — pan/zoom both panels together
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p2.line("t", "v", source=src_clean, color="seagreen", line_width=0.8)
        p2.add_tools(HoverTool(tooltips=[("Time", "@t{0.000}"), (yl, "@v{0.000000}")]))

        layout = gridplot([[p1], [p2]])

        if title:
            header = Div(
                text=f"<b style='font-size:{self._cfg.title_fontsize}'>{title}</b>",
                width=width or self._cfg.width,
            )
            return bk_column(header, layout)
        return layout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_column(df: pd.DataFrame, column: str, method: str) -> None:
        if column not in df.columns:
            raise PlotError(
                f"{method}: column '{column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )

    @staticmethod
    def _require_nonempty(values: np.ndarray, column: str, method: str) -> None:
        if len(values) == 0:
            raise PlotError(
                f"{method}: column '{column}' contains no finite values after dropping NaNs."
            )
