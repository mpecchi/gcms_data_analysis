# %%
import pytest
import pathlib as plib
from gcms_data_analysis.gcms import Project
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import shutil

folder_path: plib.Path = plib.Path(__file__).parent

folder_path = r"/Users/matteo/Projects/gcms_data_analysis/tests/data_minimal_case"
# %%
proj = Project(
    folder_path=folder_path,
    auto_save_to_excel=False,
    compounds_to_rename_in_files={"almost oleic acid": "oleic acid"},
)

# %%
files_info_created = proj.create_files_info(update_saved_files_info=False)
print(files_info_created.T)
files_info_loaded = proj.load_files_info(update_saved_files_info=False)
print(files_info_loaded.T)
# %%
files = proj.load_all_files()
print(files["S_1"])
# %%
class_code_frac = proj.load_class_code_frac()
print(class_code_frac)
# %%
calibrations = proj.load_calibrations()
print(calibrations["cal_minimal"])
# %%
list_of_all_compounds = proj.create_list_of_all_compounds()
print(list_of_all_compounds)
# %%
compounds_properties_created = proj.create_compounds_properties(
    update_saved_files_info=True
)
compounds_properties_loaded = proj.load_compounds_properties()
print(compounds_properties_created)
# %%
dict_names_to_iupac = proj.create_dict_names_to_iupacs()
print(dict_names_to_iupac)

# %%
files_iupac, calibration_iupac = proj.add_iupac_to_files_and_calibrations()
print(files_iupac["S_1"])
print(calibration_iupac["cal_minimal"])
# %%
tanimoto_similarity_df, mol_weight_diff_df = (
    proj.create_tanimoto_and_molecular_weight_similarity_dfs()
)
print(tanimoto_similarity_df)
print(mol_weight_diff_df)
# %%
semi_calibratoin_dict = proj.create_semi_calibration_dict()
print(semi_calibratoin_dict)
# %%
file1 = proj.apply_calib_to_single_file("S_1")
file2 = proj.apply_calib_to_single_file("S_2")
print(file1)
# %%
files_calibrated = proj.apply_calibration_to_files()
print(files_calibrated["S_1"])
# %%
file_info_with_stats = proj.add_stats_to_files_info()
print(file_info_with_stats)
# %%
samples_info_ave, samples_info_std = proj.create_samples_info()
print(samples_info_ave.T)
print(samples_info_std.T)
# %%
sample1_ave, sample1_std = proj.create_single_sample_from_files(
    files_in_sample=[file1, file2], samplename="S"
)
# %%
samples, samples_std = proj.create_samples_from_files()
# %%
reph = proj.create_files_param_report(param="height")
repc = proj.create_files_param_report(param="conc_vial_mg_L")
print(reph)
print(repc)
# %%
repsh, repsh_d = proj.create_samples_param_report(param="height")
repsc, repsc_d = proj.create_samples_param_report(param="conc_vial_mg_L")
print(repsh)
print(repsc)
# %%
aggh = proj.create_files_param_aggrrep(param="height")
aggc = proj.create_files_param_aggrrep(param="conc_vial_mg_L")
print(aggh)
print(aggc)
# %%
aggsh, aggsh_d = proj.create_samples_param_aggrrep(param="height")
aggsc, aggsc_d = proj.create_samples_param_aggrrep(param="conc_vial_mg_L")
print(aggsh)
print(aggsh_d)
print(aggsc)
print(aggsc_d)
# %%
proj.save_files_samples_reports()
# %%
from __future__ import annotations
from typing import Literal
from myfigure.myfigure import MyFigure, colors, hatches


def plot_ave_std(
    project: Project,
    files_or_samples: Literal["files", "samples"] = "samples",
    parameter: Literal[
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ] = "conc_vial_mg_L",
    aggregate: bool = False,
    show_total_in_twinx: bool = False,
    min_y_thresh: float | None = None,
    only_samples_to_plot: list[str] | None = None,
    rename_samples: list[str] | None = None,
    reorder_samples: list[str] | None = None,
    item_to_color_to_hatch: pd.DataFrame | None = None,
    yt_sum_label: str = "total\n(right axis)",
    **kwargs,
) -> MyFigure:
    """ """
    if show_total_in_twinx:
        plot_twinx: bool = True
    else:
        plot_twinx: bool = None
    default_kwargs = {
        "filename": "plot" + parameter,
        "out_path": proj.out_path,
        "height": 4,
        "width": 4,
        "grid": proj.plot_grid,
        "text_font": proj.plot_font,
        "y_lab": project.parameter_to_axis_label[parameter],
        "yt_lab": project.parameter_to_axis_label[parameter],
        "twinx": plot_twinx,
        "masked_unsignificant_data": True,
        # "legend": False,
    }
    # Update kwargs with the default key-value pairs if the key is not present in kwargs
    kwargs = {**default_kwargs, **kwargs}
    # create folder where Plots are stored
    out_path = plib.Path(project.out_path, "plots", files_or_samples)
    out_path.mkdir(parents=True, exist_ok=True)
    if not aggregate:  # then use compounds reports
        if files_or_samples == "files":
            df_ave = proj.files_reports[parameter].T
            df_std = pd.DataFrame()
        elif files_or_samples == "samples":
            df_ave = proj.samples_reports[parameter].T
            df_std = proj.samples_reports_std[parameter].T
    else:  # use aggregated reports
        if files_or_samples == "files":
            df_ave = proj.files_aggrreps[parameter].T
            df_std = pd.DataFrame()
        elif files_or_samples == "samples":
            df_ave = proj.samples_aggrreps[parameter].T
            df_std = proj.samples_aggrreps_std[parameter].T

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
        plot_colors = [
            item_to_color_to_hatch.loc[item, "clr"] for item in df_ave.columns
        ]
        plot_hatches = [
            item_to_color_to_hatch.loc[item, "htch"] for item in df_ave.columns
        ]
    else:  # no specific colors and hatches specified
        plot_colors = colors
        plot_hatches = hatches

    myfig = MyFigure(
        rows=1,
        cols=1,
        **kwargs,
    )
    if df_std.isna().all().all() or df_std.empty:  # means that no std is provided
        df_ave.plot(
            ax=myfig.axs[0],
            kind="bar",
            width=0.9,
            edgecolor="k",
            legend=False,
            capsize=3,
            color=colors,
        )
    else:  # no legend is represented but non-significant values are shaded
        mask = (df_ave.abs() > df_std.abs()) | df_std.isna()
        df_ave[mask].plot(
            ax=myfig.axs[0],
            kind="bar",
            width=0.9,
            edgecolor="k",
            legend=False,
            yerr=df_std[mask],
            capsize=3,
            color=colors,
            label="_nolegend_",
        )

        df_ave[~mask].plot(
            ax=myfig.axs[0],
            kind="bar",
            width=0.9,
            legend=False,
            edgecolor="grey",
            color=colors,
            alpha=0.5,
            label="_nolegend_",
        )
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
                label="_nolegend_",
            )

    myfig.save_figure()
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
    **kwargs,
) -> MyFigure:

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
            hnd_axt, lab_axt = myfig.axts[0].get_legend_handles_labels()
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


# %%


@pytest.fixture
def project():
    test_project = Project(
        folder_path=folder_path,
        auto_save_to_excel=False,
        compounds_to_rename_in_files={"almost oleic acid": "oleic acid"},
    )
    return test_project


# Test default parameters
def test_default_parameters(project):
    assert proj.column_to_sort_values_in_samples == "retention_time"
    assert proj.delta_mol_weight_threshold == 100
    assert proj.acceptable_params == [
        "height",
        "area",
        "area_if_undiluted",
        "conc_vial_mg_L",
        "conc_vial_if_undiluted_mg_L",
        "fraction_of_sample_fr",
        "fraction_of_feedstock_fr",
    ]
    assert proj.compounds_to_rename_in_files == {"almost oleic acid": "oleic acid"}


# Test the `load_files_info` method
def test_load_files_info(project):
    files_info = proj.load_files_info()
    assert isinstance(files_info, pd.DataFrame)
    assert len(files_info) > 0


# Test the `load_all_files` method
def test_load_all_files(project):
    files = proj.load_all_files()
    assert isinstance(files, dict)
    assert len(files) > 0


# Test the `load_class_code_frac` method
def test_load_class_code_frac(project):
    class_code_frac = proj.load_class_code_frac()
    assert isinstance(class_code_frac, pd.DataFrame)
    assert len(class_code_frac) > 0


# Test the `load_calibrations` method
def test_load_calibrations(project):
    calibrations = proj.load_calibrations()
    assert isinstance(calibrations, dict)
    assert len(calibrations) > 0


# Test the `create_list_of_all_compounds` method
def test_create_list_of_all_compounds(project):
    compounds = proj.create_list_of_all_compounds()
    assert isinstance(compounds, list)
    assert len(compounds) > 0


# Test the `create_compounds_properties` method
def test_create_compounds_properties(project):
    compounds_properties = proj.create_compounds_properties()
    assert isinstance(compounds_properties, pd.DataFrame)
    assert len(compounds_properties) > 0


assert_frame_equal(
    compounds_properties_created,
    compounds_properties_loaded,
    check_exact=False,
    atol=1e-5,
    rtol=1e-5,
    check_dtype=False,
)


# Test the `create_dict_names_to_iupacs` method
def test_create_dict_names_to_iupacs(project):
    dict_names_to_iupacs = proj.create_dict_names_to_iupacs()
    assert isinstance(dict_names_to_iupacs, dict)
    assert len(dict_names_to_iupacs) > 0


# Test the `add_iupac_to_files_and_calibrations` method
def test_add_iupac_to_files_and_calibrations(project):
    files_iupac, calibration_iupac = proj.add_iupac_to_files_and_calibrations()
    assert isinstance(files_iupac, dict)
    assert isinstance(calibration_iupac, dict)
    assert len(files_iupac) > 0
    assert len(calibration_iupac) > 0


# Test the `create_tanimoto_and_molecular_weight_similarity_dfs` method
def test_create_tanimoto_and_molecular_weight_similarity_dfs(project):

    tanimoto_df, mw_similarity_df = (
        proj.create_tanimoto_and_molecular_weight_similarity_dfs()
    )
    assert isinstance(tanimoto_df, pd.DataFrame)
    assert isinstance(mw_similarity_df, pd.DataFrame)
    assert len(tanimoto_df) > 0
    assert len(mw_similarity_df) > 0


# Test the `apply_calib_to_single_file` method
def test_apply_calib_to_single_file(project):
    file_name = "S_1"
    calibrated_file = proj.apply_calib_to_single_file(file_name)
    assert isinstance(calibrated_file, pd.DataFrame)
    assert len(calibrated_file) > 0


# Test the `apply_calibration_to_files` method
def test_apply_calibration_to_files(project):
    calibrated_files = proj.apply_calibration_to_files()
    assert isinstance(calibrated_files, dict)
    assert len(calibrated_files) > 0


# Test the `add_stats_to_files_info` method
def test_add_stats_to_files_info(project):
    files_info_with_stats = proj.add_stats_to_files_info()
    assert isinstance(files_info_with_stats, pd.DataFrame)
    assert len(files_info_with_stats) > 0


# Test the `create_samples_info` method
def test_create_samples_info(project):
    samples_info, samples_info_std = proj.create_samples_info()
    assert isinstance(samples_info, pd.DataFrame)
    assert isinstance(samples_info_std, pd.DataFrame)
    assert len(samples_info) > 0
    assert len(samples_info_std) > 0


# Test the `create_single_sample_from_files` method
def test_create_single_sample_from_files(project):
    files_in_sample = ["S_1", "S_2"]
    sample_name = "S"
    single_sample = proj.create_single_sample_from_files(files_in_sample, sample_name)
    assert isinstance(single_sample, pd.DataFrame)
    assert len(single_sample) > 0


# Test the `create_samples_from_files` method
def test_create_samples_from_files(project):
    samples, samples_std = proj.create_samples_from_files()
    assert isinstance(samples, dict)
    assert isinstance(samples_std, dict)
    assert len(samples) > 0
    assert len(samples_std) > 0


# Test the `create_files_param_report` method
def test_create_files_param_report(project):
    param = "height"
    files_param_report = proj.create_files_param_report(param)
    assert isinstance(files_param_report, pd.DataFrame)
    assert len(files_param_report) > 0


# Test the `create_samples_param_report` method
def test_create_samples_param_report(project):
    param = "height"
    samples_param_report, samples_param_report_std = proj.create_samples_param_report(
        param
    )
    assert isinstance(samples_param_report, pd.DataFrame)
    assert isinstance(samples_param_report_std, pd.DataFrame)
    assert len(samples_param_report) > 0
    assert len(samples_param_report_std) > 0


# Test the `create_files_param_aggrrep` method
def test_create_files_param_aggrrep(project):
    param = "height"
    files_param_aggrrep = proj.create_files_param_aggrrep(param)
    assert isinstance(files_param_aggrrep, pd.DataFrame)
    assert len(files_param_aggrrep) > 0


# Test the `create_samples_param_aggrrep` method
def test_create_samples_param_aggrrep(project):
    param = "height"
    samples_param_aggrrep, samples_param_aggrrep_std = (
        proj.create_samples_param_aggrrep(param)
    )
    assert isinstance(samples_param_aggrrep, pd.DataFrame)
    assert isinstance(samples_param_aggrrep_std, pd.DataFrame)
    assert len(samples_param_aggrrep) > 0
    assert len(samples_param_aggrrep_std) > 0


# Test the `save_files_samples_reports` method
def test_save_files_samples_reports(project):
    proj.save_files_samples_reports()
    # Add assertions to check if the reports are saved successfully
    # Test the default parameters
    for subfolder in [
        "",
        "files",
        "samples",
        "files_reports",
        "files_aggrreps",
        "samples_reports",
        "samples_aggrreps",
    ]:
        assert (folder_path / "output" / subfolder).exists()
        assert len(list((folder_path / "output" / subfolder).iterdir())) > 0

    # Remove the output folder
    # shutil.rmtree(folder_path / "output")


# check that the average between s1 and s2 is the same as s_ave for area
for param in proj.acceptable_params:
    print(f"{param = }")
    for compound in s1.index.drop("notvalidcomp").drop("dichlorobenzene"):
        print(f"\t {compound = }")

        if compound not in s2.index:
            assert np.isclose(
                s_ave.loc[compound, param],
                (s1.loc[compound, param] + 0) / 2,
            )
        else:
            assert np.isclose(
                s_ave.loc[compound, param],
                (s1.loc[compound, param] + s2.loc[compound, param]) / 2,
            )
    # do the same for the standard deviation

    for compound in s1.index.drop("notvalidcomp").drop("dichlorobenzene"):
        if compound not in s2.index:
            assert np.isclose(
                s_std.loc[compound, param], np.std((s1.loc[compound, param], 0), ddof=1)
            )
        else:
            assert np.isclose(
                s_std.loc[compound, param],
                np.std((s1.loc[compound, param], s2.loc[compound, param]), ddof=1),
            )

# Test that for each file and parameter, values match with the original file in the reports

for param in proj.acceptable_params:
    print(f"{param=}")
    rep = proj.create_files_param_report(param)
    for filename, file in files.items():
        print(f"\t{filename=}")
        for compound in file.index:
            print(f"\t\t{compound=}")
            original_values = file.loc[compound, param]
            try:
                report_values = rep.loc[compound, filename]
                assert np.allclose(original_values, report_values)
            except KeyError:
                assert np.isnan(original_values) or original_values == 0
# %%
for param in proj.acceptable_params:
    print(f"{param=}")
    rep, rep_std = proj.create_samples_param_report(param)
    for samplename, sample in samples.items():
        sample_std = proj.samples_std[samplename]
        print(f"\t{samplename=}")
        for compound in sample.index:
            print(f"\t\t{compound=}")
            original_values = sample.loc[compound, param]
            original_values_std = sample_std.loc[compound, param]
            try:
                report_values = rep.loc[compound, samplename]
                report_values_std = rep_std.loc[compound, samplename]
                assert np.allclose(original_values, report_values)
                assert np.allclose(original_values_std, report_values_std)
            except KeyError:
                assert np.isnan(original_values) or original_values == 0


# %%
