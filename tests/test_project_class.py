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
# %%
fc = proj.apply_calibration_to_files()
# %%
file_info_with_stats = proj.add_stats_to_files_info()

# %%
si_ave, si_std = proj.create_samples_info()
# %%
