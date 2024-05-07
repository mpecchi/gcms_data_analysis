# %%
import pytest
import pathlib as plib
from gcms_data_analysis.gcms import Project
from pandas.testing import assert_frame_equal
import numpy as np

folder_path: plib.Path = plib.Path(__file__).parent

folder_path = r"/Users/matteo/Projects/gcms_data_analysis/tests/data_minimal_case"
# %%
proj = Project(
    folder_path=folder_path,
    auto_save_to_excel=False,
    compounds_to_rename_in_files={"almost oleic acid": "oleic acid"},
)

# check a couple of defaults
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
# %%
fic = proj.create_files_info(update_saved_files_info=False)
fil = proj.load_files_info(update_saved_files_info=False)
import pytest
from gcms_data_analysis.gcms import Project

fic.calibration_file = fil.calibration_file  # this cannot be updated automatically
# assert_frame_equal(fil, fic, check_exact=False, atol=1e-5, rtol=1e-5)

# %%
files = proj.load_all_files()
# %%
ccf = proj.load_class_code_frac()

# %%
cal = proj.load_calibrations()
# %%
lac = proj.create_list_of_all_compounds()

# %%
cpc = proj.create_compounds_properties(update_saved_files_info=True)
cpl = proj.load_compounds_properties()
assert_frame_equal(cpc, cpl, check_exact=False, atol=1e-5, rtol=1e-5, check_dtype=False)
# %%
dni = proj.create_dict_names_to_iupacs()
assert "oleic acid" in dni.keys()
assert "notvalidcomp" in dni.keys()
assert "decanoic acid" in dni.values()
# %%
files_iupac, calibration_iupac = proj.add_iupac_to_files_and_calibrations()
# %%
tsdf, mwddf = proj.create_tanimoto_and_molecular_weight_similarity_dfs()

scd = proj.create_semi_calibration_dict()

# %%
s1 = proj.apply_calib_to_single_file("S_1")
s2 = proj.apply_calib_to_single_file("S_2")
# %%
fc = proj.apply_calibration_to_files()
# %%
file_info_with_stats = proj.add_stats_to_files_info()

# %%
si_ave, si_std = proj.create_samples_info()
# %%
s_ave, s_std = proj.create_single_sample_from_files(
    files_in_sample=[s1, s2], samplename="S"
)
# check that the average between s1 and s2 is the same as s_ave for area
for param in proj.acceptable_params:
    print(f"{param = }")
    for compound in s1.index.drop("notvalidcomp").drop("dichlorobenzene"):
        print(f"\t {compound = }")

        # print(f"\t\t {s1.loc[compound, param] = }")
        # print(f"\t\t {s2.loc[compound, param] = }")
        # print(f"\t\t {s_ave.loc[compound, param] = }")
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

# %%

# %%
samples, samples_std = proj.create_samples_from_files()

# %%
reph = proj.create_files_param_report(param="height")
repc = proj.create_files_param_report(param="conc_vial_mg_L")
# %%
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

aggsh, aggsh_d = proj.create_samples_param_aggrrep(param="height")
aggsc, aggsc_d = proj.create_samples_param_aggrrep(param="conc_vial_mg_L")
print(aggsh)
print(aggsh_d)
print(aggsc)
print(aggsc_d)
# %%
proj.save_files_samples_reports()
