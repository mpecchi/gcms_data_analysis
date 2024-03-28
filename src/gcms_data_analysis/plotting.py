from __future__ import annotations
from typing import Literal, Any, Dict
import string
import pathlib as plib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.transforms import blended_transform_factory
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
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


class MyFigure:
    """
    A class for creating and customizing figures using matplotlib and seaborn.

    MyFigure provides a structured way to create figures with multiple subplots,
    allowing for detailed customization of each subplot. It supports features like
    adjusting axis limits, adding legends, annotating, and creating inset plots,
    all with an emphasis on easy configurability through keyword arguments.

    :ivar broad_props: A dictionary to store properties that are broadcasted across all axes.
    :type broad_props: dict
    :ivar kwargs: A dictionary to store all the configuration keyword arguments.
    :type kwargs: dict
    :ivar fig: The main figure object from matplotlib.
    :type fig: matplotlib.figure.Figure
    :ivar axs: A list of axes objects corresponding to the subplots in the figure.
    :type axs: list[matplotlib.axes.Axes]
    :ivar axts: A list of twin axes objects if 'twinx' is enabled, otherwise None.
    :type axts: list[matplotlib.axes.Axes] or None
    :ivar n_axs: The number of axes/subplots in the figure.
    :type n_axs: int

    The class is designed to work seamlessly with seaborn's styling features,
    making it suitable for creating publication-quality figures with minimal code.
    """

    @staticmethod
    def _adjust_lims(lims: tuple[float] | None, gap=0.05) -> tuple[float] | None:
        """
        Adjusts the provided axis limits by a specified gap percentage to add padding
        around the data.

        :param lims: _description_
        :type lims: tuple[float] | None
        :param gap: _description_, defaults to 0.05
        :type gap: float, optional
        :return: _description_
        :rtype: tuple[float] | None
        """
        if lims is None:
            return None
        else:
            new_lims = (
                lims[0] * (1 + gap) - gap * lims[1],
                lims[1] * (1 + gap) - gap * lims[0],
            )
            return new_lims

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes a MyFigure object with custom or default settings for creating plots.

        :param kwargs: Keyword arguments to override default figure settings.
        """
        self.broad_props: dict[str, list] = {}  # broadcasted properties for each axis
        self.kwargs = self.default_kwargs()
        self.kwargs.update(kwargs)  # Override defaults with any kwargs provided
        self.process_kwargs()

        sns.set_palette(self.kwargs["color_palette"])
        sns.set_style(
            self.kwargs["sns_style"], {"font.family": self.kwargs["text_font"]}
        )

        self.create_figure()

        self.update_axes_single_props()

        self.update_axes_list_props()

    def default_kwargs(self) -> Dict[str, Any]:
        """
        Defines the default settings for the figure.

        :return: A dictionary of default settings.
        """
        defaults = {
            "rows": 1,
            "cols": 1,
            "width": 6.0,
            "height": 6.0,
            "x_lab": None,
            "y_lab": None,
            "x_lim": None,
            "y_lim": None,
            "x_ticks": None,
            "y_ticks": None,
            "x_ticklabels": None,
            "y_ticklabels": None,
            "twinx": False,
            "yt_lab": None,
            "yt_lim": None,
            "yt_ticks": None,
            "yt_ticklabels": None,
            "legend": True,
            "legend_loc": "best",
            "legend_ncols": 1,
            "annotate_lttrs": False,
            "annotate_lttrs_xy": None,
            "grid": False,
            "color_palette": "deep",
            "text_font": "Dejavu Sans",
            "sns_style": "ticks",
        }
        return defaults

    def process_kwargs(self) -> None:
        """
        Validates and processes the provided keyword arguments for figure configuration.


        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        """
        self.kwargs["rows"] = int(self.kwargs["rows"])
        self.kwargs["cols"] = int(self.kwargs["cols"])
        self.kwargs["width"] = float(self.kwargs["width"])
        self.kwargs["height"] = float(self.kwargs["height"])
        self.kwargs["legend_ncols"] = int(self.kwargs["legend_ncols"])

        if self.kwargs["rows"] <= 0:
            raise ValueError("Number of rows must be positive.")
        if self.kwargs["cols"] <= 0:
            raise ValueError("Number of cols must be positive.")
        if self.kwargs["width"] <= 0:
            raise ValueError("Width must be positive.")
        if self.kwargs["height"] <= 0:
            raise ValueError("Height must be positive.")
        if self.kwargs["legend_ncols"] <= 0:
            raise ValueError("Number of legend columns must be positive.")

    def create_figure(self) -> MyFigure:
        """
        Creates the figure and its axes.

        :return: _description_
        :rtype: MyFigure
        """
        self.fig: Figure
        self.axs: Axes
        self.axts: Axes | None
        self.fig, axes = plt.subplots(
            self.kwargs["rows"],
            self.kwargs["cols"],
            figsize=(self.kwargs["width"], self.kwargs["height"]),
            constrained_layout=True,
        )
        # Ensure ax is always an array, even if it's just one subplot
        self.axs: list[Axes] = np.atleast_1d(axes).flatten().tolist()
        if self.kwargs["twinx"]:
            self.axts: list[Axes] = [a.twinx() for a in self.axs]

        self.n_axs = len(self.axs)
        return self

    def save_figure(
        self,
        filename: str = "figure",
        out_path: plib.Path | None = plib.Path("."),
        tight_layout: bool = True,
        save_as_png: bool = True,
        save_as_pdf: bool = False,
        save_as_svg: bool = False,
        save_as_eps: bool = False,
        png_transparency: bool = False,
    ) -> None:
        """_summary_

        :param filename: _description_, defaults to "figure"
        :type filename: str, optional
        :param out_path: _description_, defaults to plib.Path(".")
        :type out_path: plib.Path | None, optional
        :param tight_layout: _description_, defaults to True
        :type tight_layout: bool, optional
        :param save_as_png: _description_, defaults to True
        :type save_as_png: bool, optional
        :param save_as_pdf: _description_, defaults to False
        :type save_as_pdf: bool, optional
        :param save_as_svg: _description_, defaults to False
        :type save_as_svg: bool, optional
        :param save_as_eps: _description_, defaults to False
        :type save_as_eps: bool, optional
        :param png_transparency: _description_, defaults to False
        :type png_transparency: bool, optional
        """
        self.update_axes_single_props()

        self.update_axes_list_props()

        self.add_legend()
        try:
            self.fig.align_labels()  # align labels of subplots, needed only for multi plot
        except AttributeError:
            print("align_labels not performed")
        self.annotate_letters()
        # Saving the figure
        formats = {
            "png": save_as_png,
            "pdf": save_as_pdf,
            "svg": save_as_svg,
            "eps": save_as_eps,
        }

        for fmt, should_save in formats.items():
            if should_save:
                full_path = plib.Path(out_path, f"{filename}.{fmt}")
                self.fig.savefig(
                    full_path,
                    dpi=300,
                    transparent=png_transparency,
                    bbox_inches="tight" if tight_layout else None,
                )

    def add_legend(self) -> None:
        """_summary_"""
        for sprop in ["legend", "legend_loc", "legend_ncols"]:
            self.broad_props[sprop] = self._broadcast_value_prop(
                self.kwargs[sprop], sprop
            )

        if self.kwargs["twinx"] is False:
            for i, ax in enumerate(self.axs):
                if self.broad_props["legend"][i]:
                    ax.legend(
                        loc=self.broad_props["legend_loc"][i],
                        ncol=self.broad_props["legend_ncols"][i],
                    )
        else:
            for i, (ax, axt) in enumerate(zip(self.axs, self.axts)):
                if self.broad_props["legend"][i]:
                    hnd_ax, lab_ax = ax.get_legend_handles_labels()
                    hnd_axt, lab_axt = axt.get_legend_handles_labels()
                    ax.legend(
                        hnd_ax + hnd_axt,
                        lab_ax + lab_axt,
                        loc=self.broad_props["legend_loc"][i],
                        ncol=self.broad_props["legend_ncols"][i],
                    )

    def annotate_letters(self) -> None:
        """_summary_"""
        if (
            self.kwargs["annotate_lttrs_xy"] is not None
            and isinstance(self.kwargs["annotate_lttrs_xy"], (list, tuple))
            and len(self.kwargs["annotate_lttrs_xy"]) >= 2
        ):
            xylttrs: list | tuple = self.kwargs["annotate_lttrs_xy"]
            x_lttrs = xylttrs[0]  # pylint: disable=unsubscriptable-object
            y_lttrs = xylttrs[1]  # pylint: disable=unsubscriptable-object
        else:
            x_lttrs = -0.15
            y_lttrs = -0.15
        if self.kwargs["annotate_lttrs"] is not False:
            if isinstance(self.kwargs["annotate_lttrs"], str):
                letters_list = [self.kwargs["annotate_lttrs"]]
            elif isinstance(self.kwargs["annotate_lttrs"], list, tuple):
                letters_list = self.kwargs["annotate_lttrs"]
            for i, ax in enumerate(self.axs):
                ax.annotate(
                    f"({letters_list[i]})",
                    xycoords="axes fraction",
                    xy=(0, 0),
                    xytext=(x_lttrs, y_lttrs),
                    size="large",
                    weight="bold",
                )

    def create_inset(
        self,
        ax: Axes,
        ins_x_loc: list[float, float],
        ins_y_loc: list[float, float],
        ins_x_lim: list[float, float],
        ins_y_lim: list[float, float],
    ) -> Axes:
        """_summary_

        :param ax: _description_
        :type ax: Axes
        :param ins_x_loc: _description_
        :type ins_x_loc: list[float, float]
        :param ins_y_loc: _description_
        :type ins_y_loc: list[float, float]
        :param ins_x_lim: _description_
        :type ins_x_lim: list[float, float]
        :param ins_y_lim: _description_
        :type ins_y_lim: list[float, float]
        :return: _description_
        :rtype: Axes
        """
        wdt = ins_x_loc[1] - ins_x_loc[0]
        hgt = ins_y_loc[1] - ins_y_loc[0]
        inset = ax.inset_axes([ins_x_loc[0], ins_y_loc[0], wdt, hgt])

        inset.set_xlim(MyFigure._adjust_lims(ins_x_lim))
        inset.set_ylim(MyFigure._adjust_lims(ins_y_lim))
        return inset

    def update_axes_single_props(self):
        """_summary_"""
        for sprop in ["x_lab", "y_lab", "yt_lab", "grid"]:
            self.broad_props[sprop] = self._broadcast_value_prop(
                self.kwargs[sprop], sprop
            )

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            ax.set_xlabel(self.broad_props["x_lab"][i])
            ax.set_ylabel(self.broad_props["y_lab"][i])
            if self.broad_props["grid"][i] is not None:
                ax.grid(self.broad_props["grid"][i])

        if self.kwargs["twinx"]:
            for i, axt in enumerate(self.axts):
                axt.set_ylabel(self.broad_props["yt_lab"][i])

    def update_axes_list_props(self):
        """_summary_"""
        for lprop in [
            "x_lim",
            "y_lim",
            "yt_lim",
            "x_ticks",
            "y_ticks",
            "yt_ticks",
            "x_ticklabels",
            "y_ticklabels",
            "yt_ticklabels",
        ]:
            self.broad_props[lprop] = self._broadcast_list_prop(
                self.kwargs[lprop], lprop
            )

        # Update each axis with the respective properties
        for i, ax in enumerate(self.axs):
            if self.broad_props["x_lim"][i] is not None:
                ax.set_xlim(MyFigure._adjust_lims(self.broad_props["x_lim"][i]))
            if self.broad_props["y_lim"][i] is not None:
                ax.set_ylim(MyFigure._adjust_lims(self.broad_props["y_lim"][i]))
            if self.broad_props["x_ticks"][i] is not None:
                ax.set_xticks(self.broad_props["x_ticks"][i])
            if self.broad_props["y_ticks"][i] is not None:
                ax.set_yticks(self.broad_props["y_ticks"][i])
            if self.broad_props["x_ticklabels"][i] is not None:
                ax.set_xticklabels(self.broad_props["x_ticklabels"][i])
            if self.broad_props["y_ticklabels"][i] is not None:
                ax.set_yticklabels(self.broad_props["y_ticklabels"][i])

        if self.kwargs["twinx"]:
            for i, axt in enumerate(self.axts):
                if self.broad_props["yt_lim"][i] is not None:
                    axt.set_ylim(MyFigure._adjust_lims(self.broad_props["yt_lim"][i]))
                if self.broad_props["yt_ticks"][i] is not None:
                    axt.set_yticks(self.broad_props["yt_ticks"][i])
                if self.broad_props["yt_ticklabels"][i] is not None:
                    axt.set_yticklabels(self.broad_props["yt_ticklabels"][i])

    def _broadcast_value_prop(
        self, prop: list | str | float | int | bool, prop_name: str
    ) -> list:
        """_summary_

        :param prop: _description_
        :type prop: list | str | float | int | bool
        :param prop_name: The name of the property for error messages.
        :type prop_name: str
        :raises ValueError: _description_
        :return: _description_
        :rtype: list
        """
        if prop is None:
            return [None] * self.n_axs
        if isinstance(prop, (list, tuple)):
            if len(prop) == self.n_axs:
                return prop
            else:
                raise ValueError(
                    f"The size of the property '{prop_name}' does not match the number of axes."
                )
        if isinstance(prop, (str, float, int, bool)):
            return [prop] * self.n_axs

    def _broadcast_list_prop(self, prop: list | None, prop_name: str):
        """_summary_

        :param prop: _description_
        :type prop: list | None
        :param prop_name: The name of the property for error messages.
        :type prop_name: str
        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """
        if prop is None:
            return [None] * self.n_axs

        if (
            all(isinstance(item, (list, tuple)) for item in prop)
            and len(prop) == self.n_axs
        ):
            return prop
        elif isinstance(prop, (list, tuple)) and all(
            isinstance(item, (int, float, str)) for item in prop
        ):
            return [prop] * self.n_axs
        else:
            raise ValueError(
                f"The structure of '{prop_name = }' does not match expected pair-wise input."
            )


def plot_ave_std(
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
            hnd_axt, lab_axt = myfig.axt[0].get_legend_handles_labels()
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


def plot_df_ave_std(
    proj: Project,
    df_ave: pd.DataFrame,
    df_std: pd.DataFrame = pd.DataFrame(),
    filename: str = "plot",
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
    out_path = plib.Path(Project.out_path, "df_plots")
    out_path.mkdir(parents=True, exist_ok=True)
    if only_samples_to_plot is not None:
        df_ave = df_ave.loc[only_samples_to_plot, :].copy()
        if not df_std.empty:
            df_std = df_std.loc[only_samples_to_plot, :].copy()

    if rename_samples is not None:
        df_ave.index = rename_samples
        if not df_std.empty:
            df_std.index = rename_samples

    if reorder_samples is not None:
        filtered_reorder_samples = [
            idx for idx in reorder_samples if idx in df_ave.index
        ]
        df_ave = df_ave.reindex(filtered_reorder_samples)
        if not df_std.empty:
            df_std = df_std.reindex(filtered_reorder_samples)
    if reorder_samples is not None:
        filtered_reorder_samples = [
            idx for idx in reorder_samples if idx in df_ave.index
        ]
        df_ave = df_ave.reindex(filtered_reorder_samples)
        if not df_std.empty:
            df_std = df_std.reindex(filtered_reorder_samples)

    if min_y_thresh is not None:
        df_ave = df_ave.loc[:, (df_ave > min_y_thresh).any(axis=0)].copy()
        if not df_std.empty:
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
            hnd_axt, lab_axt = myfig.axt[0].get_legend_handles_labels()
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
