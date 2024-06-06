# %% Import necessary libraries
import pathlib as plib
from gcms_data_analysis.gcms import Project

folder_path: plib.Path = plib.Path(__file__).parent

folder_path = r"/Users/matteo/Projects/gcms_data_analysis/examples/data_minimal_case"

proj = Project(
    folder_path=folder_path,
    auto_save_to_excel=False,
    compounds_to_rename_in_files={
        "almost oleic acid": "oleic acid",
        "dichlorobenzene": "p-dichlorobenzene",
    },
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
print(reph)

repc = proj.create_files_param_report(param="conc_vial_mg_L")
print(repc)
# %%
repsh, repsh_d = proj.create_samples_param_report(param="height")
print(repsh)
repsc, repsc_d = proj.create_samples_param_report(param="conc_vial_mg_L")
print(repsc)
# %%
aggh = proj.create_files_param_aggrrep(param="height")
print(aggh)
# %%
aggc = proj.create_files_param_aggrrep(param="conc_vial_mg_L")

print(aggc)
# %%
aggsh, aggsh_d = proj.create_samples_param_aggrrep(param="height")
print(aggsh)
print(aggsh_d)
# %%
aggsc, aggsc_d = proj.create_samples_param_aggrrep(param="conc_vial_mg_L")

print(aggsc)
print(aggsc_d)
# %%
proj.save_files_samples_reports()
# %%

proj.plot_report()

# %%
