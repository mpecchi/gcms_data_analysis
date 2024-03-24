from __future__ import annotations
from typing import Literal
import string
import pathlib as plib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.transforms import blended_transform_factory
from gcms_data_analysis.my_figure import MyFigure
from gcms_data_analysis.main import Project

lttrs: list[str] = list(string.ascii_lowercase)

# list with colors
clrs: list[tuple] = sns.color_palette("deep", 30)

# list with linestyles for plotting
lnstls: list[tuple] = [
    (0, ()),  # solid
    (0, (1, 1)),  # 'densely dotted'
    (0, (5, 1)),  # 'densely dashed'
    (0, (3, 1, 1, 1)),  # 'densely dashdotted'
    (0, (3, 1, 1, 1, 1, 1)),  # 'densely dashdotdotted'
    (0, (5, 5)),  # 'dashed'
    (0, (3, 5, 1, 5)),  # 'dashdotted'
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5, 1, 5)),  # 'dashdotdotted'
    (0, (1, 10)),  # 'loosely dotted'
    (0, (5, 10)),  # 'loosely dashed'
    (0, (3, 10, 1, 10)),  # 'loosely dashdotted'
    (0, (3, 10, 1, 10, 1, 10)),
    (0, ()),  # solid
    (0, (1, 1)),  # 'densely dotted'
    (0, (5, 1)),  # 'densely dashed'
    (0, (3, 1, 1, 1)),  # 'densely dashdotted'
    (0, (3, 1, 1, 1, 1, 1)),  # 'densely dashdotdotted'
    (0, (5, 5)),  # 'dashed'
    (0, (3, 5, 1, 5)),  # 'dashdotted'
    (0, (1, 5)),  # dotted
    (0, (3, 5, 1, 5, 1, 5)),  # 'dashdotdotted'
    (0, (1, 10)),  # 'loosely dotted'
    (0, (5, 10)),  # 'loosely dashed'
    (0, (3, 10, 1, 10)),  # 'loosely dashdotted'
    (0, (3, 10, 1, 10, 1, 10)),
]  # 'loosely dashdotdotted'

# list with markers for plotting
mrkrs: list[str] = [
    "o",
    "v",
    "X",
    "s",
    "p",
    "^",
    "P",
    "<",
    ">",
    "*",
    "d",
    "1",
    "2",
    "3",
    "o",
    "v",
    "X",
    "s",
    "p",
    "^",
    "P",
    "<",
    ">",
    "*",
    "d",
    "1",
    "2",
    "3",
]

htchs: list[str] = [
    None,
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
    "//",
    "...",
    "--",
    "O",
    "\\\\",
    "oo",
    "\\\\\\",
    "/////",
    ".....",
]


def _annotate_outliers_in_plot(ax, df_ave, df_std, y_lim):
    """
    Annotates the bars in a bar plot with their average value and standard
    deviation if these values exceed the specified y-axis limits.
    The function iterates over the bars in the plot and checks if their average
    values, considering their standard deviations, are outside the provided
    y-axis limits. For such bars, it annotates the average and standard
    deviation on the
    plot, using a specific format for better visualization and understanding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the plot is drawn.
    df_ave : pandas.DataFrame
        DataFrame containing the average values used in the plot.
    df_std : pandas.DataFrame
        DataFrame containing the standard deviation values corresponding
        to df_ave.
    y_lim : list of [float, float]
        A list of two floats representing the minimum (y_lim[0]) and
        maximum (y_lim[1]) limits of the y-axis.

    Returns
    -------
    None
        Modifies the provided Axes object (ax) by adding annotations.

    """
    dx = 0.15 * len(df_ave.index)
    dy = 0.04
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    dfao = pd.DataFrame(columns=["H/L", "xpos", "ypos", "ave", "std", "text"])
    dfao["ave"] = df_ave.transpose().to_numpy().flatten().tolist()
    if df_std.empty:
        df_std = np.zeros(len(dfao["ave"]))
    else:
        dfao["std"] = df_std.transpose().to_numpy().flatten().tolist()
    try:
        dfao["xpos"] = [p.get_x() + p.get_width() / 2 for p in ax.patches]
    except ValueError:  # otherwise the masking adds twice the columns
        dfao["xpos"] = [
            p.get_x() + p.get_width() / 2 for p in ax.patches[: len(ax.patches) // 2]
        ]
    cond = (dfao["ave"] < y_lim[0]) | (dfao["ave"] > y_lim[1])
    dfao = dfao.drop(dfao[~cond].index)
    for ao in dfao.index.tolist():  # loop through bars
        if dfao.loc[ao, "ave"] == float("inf"):
            dfao.loc[ao, "text"] = "inf"
            dfao.loc[ao, "H/L"] = "H"
        elif dfao.loc[ao, "ave"] == float("-inf"):
            dfao.loc[ao, "text"] = "-inf"
            dfao.loc[ao, "H/L"] = "L"
        elif dfao.loc[ao, "ave"] > y_lim[1]:
            dfao.loc[ao, "H/L"] = "H"
            dfao.loc[ao, "text"] = "{:.2f}".format(
                round(dfao.loc[ao, "ave"], 2)
            ).strip()
            if (dfao.loc[ao, "std"] != 0) & (~np.isnan(dfao.loc[ao, "std"])):
                dfao.loc[ao, "text"] += r"$\pm$" + "{:.2f}".format(
                    round(dfao.loc[ao, "std"], 2)
                )
        elif dfao.loc[ao, "ave"] < y_lim[0]:
            dfao.loc[ao, "H/L"] = "L"
            dfao.loc[ao, "text"] = str(round(dfao.loc[ao, "ave"], 2)).strip()
            if dfao.loc[ao, "std"] != 0:
                dfao.loc[ao, "text"] += r"$\pm$" + "{:.2f}".format(
                    round(dfao.loc[ao, "std"], 2)
                )
        else:
            print("Something is wrong", dfao.loc[ao, "ave"])
    for hl, ypos, dy in zip(["L", "H"], [0.02, 0.98], [0.04, -0.04]):
        dfao1 = dfao[dfao["H/L"] == hl]
        dfao1["ypos"] = ypos
        if not dfao1.empty:
            dfao1 = dfao1.sort_values("xpos", ascending=True)
            dfao1["diffx"] = (
                np.diff(dfao1["xpos"].values, prepend=dfao1["xpos"].values[0]) < dx
            )
            dfao1.reset_index(inplace=True)

            for i in dfao1.index.tolist()[1:]:
                dfao1.loc[i, "ypos"] = ypos
                for e in range(i, 0, -1):
                    if dfao1.loc[e, "diffx"]:
                        dfao1.loc[e, "ypos"] += dy
                    else:
                        break
            for ao in dfao1.index.tolist():
                ax.annotate(
                    dfao1.loc[ao, "text"],
                    xy=(dfao1.loc[ao, "xpos"], 0),
                    xycoords=tform,
                    textcoords=tform,
                    xytext=(dfao1.loc[ao, "xpos"], dfao1.loc[ao, "ypos"]),
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "square,pad=0",
                        "edgecolor": None,
                        "facecolor": "white",
                        "alpha": 0.7,
                    },
                )


def plot_pave_std(
    proj: Project,
    filename: str = "plot",
    files_or_samples: Literal["files", "samples"] = "samples",
    param: str = "conc_vial_mg_L",
    aggr: bool = False,
    show_total_in_twinx: bool = False,
    annotate_outliers: bool = True,
    min_y_thresh: float | None = None,
    only_samples_to_plot: list[str] | None = None,
    rename_samples: list[str] | None = None,
    reorder_samples: list[str] | None = None,
    item_to_color_to_hatch: pd.DataFrame | None = None,
    yt_sum_label: str = "total\n(right axis)",
    y_lim: tuple[float] | None = None,
    y_lab: str | None = None,
    yt_lab: str | None = None,
    color_palette: str = "deep",
    x_label_rotation: int = 0,
    legend_location: Literal["best", "outside"] = "best",
    legend_columns: int = 1,
    legend_x_anchor: float = 1,
    legend_y_anchor: float = 1.02,
    legend_labelspacing: float = 0.5,
    **kwargs,
) -> MyFigure:
    """
    Generates a bar plot displaying average values with optional standard deviation
    bars for a specified parameter from either files or samples. This function allows
    for detailed customization of the plot, including aggregation by functional groups,
    filtering based on minimum thresholds, renaming and reordering samples, and applying
    specific color schemes and hatching patterns to items.
    Additionally, it supports adjusting plot aesthetics such as size, figure height multiplier,
    x-label rotation, and outlier annotation. The plot can include a secondary y-axis
    to display the sum of values, with customizable limits, labels, ticks, and sum label.
    The legend can be placed inside or outside the plot area, with adjustable location,
    columns, anchor points, and label spacing. An optional note can be added to the plot
    for additional context.

    Parameters:

    filename (str): Name for the output plot file. Default is 'plot'.

    files_or_samples (str): Specifies whether to plot data from 'files'
        or 'samples'. Default is 'samples'.

    param (str): The parameter to plot, such as 'conc_vial_mg_L'.
        Default is 'conc_vial_mg_L'.

    aggr (bool): Boolean indicating whether to aggregate data by functional groups.
        Default is False, meaning no aggregation.

    min_y_thresh (float, optional): Minimum y-value threshold for including data in the plot.
        Default is None, including all data.

    only_samples_to_plot (list, optional): List of samples to include in the plot.
        Default is None, including all samples.

    rename_samples (dict, optional): Dictionary to rename samples in the plot.
        Default is None, using original names.

    reorder_samples (list, optional): List specifying the order of samples in the plot.
        Default is None, using original order.

    item_to_color_to_hatch (DataFrame, optional): DataFrame mapping items to specific colors and hatching patterns.
        Default is None, using default colors and no hatching.

    paper_col (float): Background color of the plot area. Default is .8, a light grey.

    fig_hgt_mlt (float): Multiplier for the figure height to adjust plot size. Default is 1.5.

    x_label_rotation (int): Rotation angle for x-axis labels. Default is 0, meaning no rotation.

    annotate_outliers (bool): Boolean indicating whether to annotate outliers exceeding y_lim.
        Default is True.

    color_palette (str): Color palette for the plot. Default is 'deep'.

    y_lab (str, optional): Label for the y-axis. Default is None, using parameter name as label.

    y_lim (tuple[float, float], optional): Limits for the y-axis. Default is None, automatically determined.

    y_ticks (list[float], optional): Custom tick marks for the y-axis. Default is None, automatically determined.

    yt_sum (bool): Boolean indicating whether to display a sum on a secondary y-axis. Default is False.

    yt_lim (tuple[float, float], optional): Limits for the secondary y-axis. Default is None, automatically determined.

    yt_lab (str, optional): Label for the secondary y-axis. Default is None, using parameter name as label.

    yt_ticks (list[float], optional): Custom tick marks for the secondary y-axis. Default is None, automatically determined.

    yt_sum_label (str): Label for the sum on the secondary y-axis. Default is 'total (right axis)'.

    legend_location (str): Location of the legend within or outside the plot area. Default is 'best'.

    legend_columns (int): Number of columns in the legend. Default is 1.

    legend_x_anchor (float): X-anchor for the legend when placed outside the plot area. Default is 1.

    legend_y_anchor (float): Y-anchor for the legend when placed outside the plot area. Default is 1.02.

    legend_labelspacing (float): Spacing between labels in the legend. Default is 0.5.

    annotate_lttrs (bool): Boolean indicating whether to annotate letters for statistical significance. Default is False.

    note_plt (str, optional): Optional note to add to the plot for additional context. Default is None.

    """

    # create folder where Plots are stored
    out_path = plib.Path(Project.out_path, "plots")
    out_path.mkdir(parents=True, exist_ok=True)
    if not aggr:  # then use compounds reports
        if files_or_samples == "files":
            df_ave = proj.files_reports[param].T
            df_std = pd.DataFrame()
        elif files_or_samples == "samples":
            df_ave = proj.samples_reports[param].T
            df_std = proj.samples_reports_std[param].T
    else:  # use aggregated reports
        if files_or_samples == "files":
            df_ave = proj.files_aggrreps[param].T
            df_std = pd.DataFrame()
        elif files_or_samples == "samples":
            df_ave = proj.samples_aggrreps[param].T
            df_std = proj.samples_aggrreps_std[param].T

    if only_samples_to_plot is not None:
        df_ave = df_ave.loc[only_samples_to_plot, :].copy()
        if files_or_samples == "samples":
            df_std = df_std.loc[only_samples_to_plot, :].copy()

    if rename_samples is not None:
        df_ave.index = rename_samples
        if files_or_samples == "samples":
            df_std.index = rename_samples

    if reorder_samples is not None:
        filtered_reorder_samples = [
            idx for idx in reorder_samples if idx in df_ave.index
        ]
        df_ave = df_ave.reindex(filtered_reorder_samples)
        if files_or_samples == "samples":
            df_std = df_std.reindex(filtered_reorder_samples)

    if min_y_thresh is not None:
        df_ave = df_ave.loc[:, (df_ave > min_y_thresh).any(axis=0)].copy()
        if files_or_samples == "samples":
            df_std = df_std.loc[:, df_ave.columns].copy()

    if item_to_color_to_hatch is not None:  # specific color and hatches to each fg
        colors = [item_to_color_to_hatch.loc[item, "clr"] for item in df_ave.columns]
        hatches = [item_to_color_to_hatch.loc[item, "htch"] for item in df_ave.columns]
    else:  # no specific colors and hatches specified
        colors = sns.color_palette(color_palette, df_ave.shape[1])
        hatches = htchs

    if show_total_in_twinx:
        plot_twinx: bool = True
    else:
        plot_twinx: bool = False

    if y_lab is None:
        y_lab = Project.param_to_axis_label[param]
    if show_total_in_twinx:
        legend_x_anchor += 0.14
        yt_lab = y_lab

    myfig = MyFigure(
        rows=1,
        cols=1,
        twinx=plot_twinx,
        text_font=Project.plot_font,
        y_lab=y_lab,
        yt_lab=yt_lab,
        y_lim=y_lim,
        legend=False,
        grid=Project.plot_grid,
        **kwargs,
    )
    if df_std.isna().all().all() or df_std.empty:  # means that no std is provided
        df_ave.plot(
            ax=myfig.axs[0],
            kind="bar",
            rot=x_label_rotation,
            width=0.9,
            edgecolor="k",
            legend=False,
            capsize=3,
            color=colors,
        )
        bars = myfig.axs[0].patches  # needed to add patches to the bars
        n_different_hatches = int(len(bars) / df_ave.shape[0])
    else:  # no legend is represented but non-significant values are shaded
        mask = (df_ave.abs() > df_std.abs()) | df_std.isna()

        df_ave[mask].plot(
            ax=myfig.axs[0],
            kind="bar",
            rot=x_label_rotation,
            width=0.9,
            edgecolor="k",
            legend=False,
            yerr=df_std[mask],
            capsize=3,
            color=colors,
            label="_nolegend",
        )
        df_ave[~mask].plot(
            ax=myfig.axs[0],
            kind="bar",
            rot=x_label_rotation,
            width=0.9,
            legend=False,
            edgecolor="grey",
            color=colors,
            alpha=0.5,
            label="_nolegend",
        )
        bars = myfig.axs[0].patches  # needed to add patches to the bars
        n_different_hatches = int(len(bars) / df_ave.shape[0] / 2)
    if show_total_in_twinx:
        myfig.axts[0].scatter(
            df_ave.index,
            df_ave.sum(axis=1).values,
            color="k",
            linestyle="None",
            edgecolor="k",
            facecolor="grey",
            s=100,
            label=yt_sum_label,
            alpha=0.5,
        )
        if not df_std.empty:
            myfig.axts[0].errorbar(
                df_ave.index,
                df_ave.sum(axis=1).values,
                df_std.sum(axis=1).values,
                capsize=3,
                linestyle="None",
                color="grey",
                ecolor="k",
            )
    bar_hatches = []
    # get a list with the hatches
    for h in hatches[:n_different_hatches] + hatches[:n_different_hatches]:
        for n in range(df_ave.shape[0]):  # htcs repeated for samples
            bar_hatches.append(h)  # append based on samples number
    for bar, hatch in zip(bars, bar_hatches):  # assign hatches to each bar
        bar.set_hatch(hatch)
    myfig.axs[0].set(xlabel=None)
    if x_label_rotation != 0:
        myfig.axs[0].set_xticklabels(
            df_ave.index, rotation=x_label_rotation, ha="right", rotation_mode="anchor"
        )
    if legend_location is not None:
        hnd_ax, lab_ax = myfig.axs[0].get_legend_handles_labels()
        if not df_std.empty:
            hnd_ax = hnd_ax[: len(hnd_ax) // 2]
            lab_ax = lab_ax[: len(lab_ax) // 2]
        if legend_labelspacing > 0.5:  # large legend spacing for molecules
            myfig.axs[0].plot(np.nan, np.nan, "-", color="None", label=" ")
            hhhh, aaaa = myfig.axs[0].get_legend_handles_labels()
            hnd_ax.append(hhhh[0])
            lab_ax.append(aaaa[0])
        if show_total_in_twinx:
            hnd_axt, lab_axt = axt[0].get_legend_handles_labels()
        else:
            hnd_axt, lab_axt = [], []
        if legend_location == "outside":  # legend goes outside of plot area
            myfig.axs[0].legend(
                hnd_ax + hnd_axt,
                lab_ax + lab_axt,
                loc="upper left",
                ncol=legend_columns,
                bbox_to_anchor=(legend_x_anchor, legend_y_anchor),
                labelspacing=legend_labelspacing,
            )
        else:  # legend is inside of plot area
            myfig.axs[0].legend(
                hnd_ax + hnd_axt,
                lab_ax + lab_axt,
                loc=legend_location,
                ncol=legend_columns,
                labelspacing=legend_labelspacing,
            )
    # annotate ave+-std at the top of outliers bar (exceeding y_lim)
    if annotate_outliers and (y_lim is not None):  # and (not df_std.empty):
        _annotate_outliers_in_plot(myfig.axs[0], df_ave, df_std, y_lim)
    myfig.save_figure(filename, out_path)
    return myfig


# if __file__ == "__main__":
#     f = MyFigure(
#         rows=4,
#         cols=1,
#         width=6,
#         height=12,
#         twinx=True,
#         x_lab=["aaa", "qqq", "aa", "qq"],
#         y_lab="bbb",
#         yt_lab="ccc",
#         x_lim=[0, 1],
#         y_lim=[0, 1],
#         yt_lim=[[0, 1], [0, 0.5], [0, 1], [0, 0.5]],
#         x_ticks=[[0, 0.5, 1], [0, 0.5, 2], [0, 1], [0, 0.5]],
#         # x_ticklabels=["a", "c", "d"],
#         grid=True,
#         annotate_lttrs=["a", "b", "a", "b"],
#         annotate_lttrs_xy=[-0.11, -0.15],
#     )

#     f.axs[0].plot([0, 1], [0, 3], label="a")
#     f.axts[0].plot([0, 2], [0, 4], label="b")
#     f.axts[0].plot([0, 2], [0, 5], label="ccc")
#     f.axs[1].plot([0, 1], [0, 3], label="aaa")
#     ins = f.create_insex(f.axs[0], [0.6, 0.8], [0.4, 0.6], [0, 0.2], [0, 0.2])
#     ins.plot([0, 1], [0, 3], label="a")
#     f.save_figure(
#         filename="my_plot", out_path=plib.Path(r"C:\Users\mp933\Desktop\New folder")
#     )
