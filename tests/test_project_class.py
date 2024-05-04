# %%
import pytest
import pathlib as plib
from gcms_data_analysis.gcms import Project
from pandas.testing import assert_frame_equal


folder_path: plib.Path = plib.Path(__file__).parent

folder_path = r"/Users/matteo/Projects/gcms_data_analysis/tests/data_minimal_case"
# %%
proj = Project(
    folder_path=folder_path,
    auto_save_reports=False,
    compounds_to_rename_in_files={"phenol": "renamed_phenol"},
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
assert_frame_equal(fil, fic, check_exact=False, atol=1e-5, rtol=1e-5)
# print(fil.columns)
# print(fic.columns)
# print(fil.index)
# print(fic.index)
# print(fil==fic)
# %%
files = proj.load_all_files()
# %%
ccf = proj.load_class_code_frac()

# %%
cal = proj.load_calibrations()
print(cal)
# %%
lac = proj.create_list_of_all_compounds()
