from __future__ import annotations
import string
import pathlib as plib
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns


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
        if self.kwargs["annotate_lttrs_xy"] is not None and isinstance(
            self.kwargs["annotate_lttrs_xy"],
            (list, tuple) and len(self.kwargs["annotate_lttrs_xy"]) >= 2,
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
