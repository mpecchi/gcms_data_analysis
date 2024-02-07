
import pathlib as plib
from gcms_data_analysis import Project

# you might need to change this to the path were you have the data
folder_path = plib.Path(plib.Path(__file__).cwd(), 'data')

# class methods need to be called at the beginning to influence all instances
Project.set_folder_path(folder_path)
Project.set_folder_path(folder_path)
Project.set_plot_grid(False)
Project.set_plot_font('Sans')  # ('Times New Roman')
# initialize project
p = Project()
# load files_info as provided by the user, if not given, create it
# using the GC-MS .txt files in the folder
files_info_0 = p.load_files_info()
# load the provided calibrations as dict, store bool to know if are deriv
calibrations, is_calibr_deriv = p.load_calibrations()
c1, c2 = calibrations['CalDB'], calibrations['CalDBder']
# load provided classificaiton codes and mass fractions for fun. groups
class_code_frac = p.load_class_code_frac()
# load all GCMS txt files as single files
files0, is_files_deriv0 = p.load_all_files()
f1, f2, f3 = files0['A_1'], files0['Ader_1'], files0['B_1']
# create the list with all compounds in all samples
list_of_all_compounds = p.create_list_of_all_compounds()
# create the list with all derivatized compounds in all samples
list_of_all_deriv_compounds = p.create_list_of_all_deriv_compounds()
if 0: # set 0 to 1 if you want to recreate compounds properties databases
    compounds_properties = p.create_compounds_properties()
    deriv_compounds_properties = p.create_deriv_compounds_properties()
else:  # otherwise load the available one (if unavailable it creates them)
    compounds_properties = p.load_compounds_properties()
    deriv_compounds_properties = p.load_deriv_compounds_properties()
# apply the calibration to all files and store updated files as dict
files, is_files_deriv = p.apply_calibration_to_files()
f11, f22, f33 = files['A_1'], files['Ader_1'], files['B_1']
# compute stats for each file in the files_info df
files_info = p.add_stats_to_files_info()
# create samples_info (ave and std) based on replicate data in files_info
samples_info_0 = p.create_samples_info()
# create samples and samples_std from files and store as dict
samples, samples_std = p.create_samples_from_files()
s1, s2, s3 = samples['A'], samples['Ader'], samples['B']
sd1, sd2, sd3 = samples_std['A'], samples_std['Ader'], samples_std['B']
# add stats to samples_info df
samples_info = p.add_stats_to_samples_info()
# create report (compounds based) for different parameters
rep_conc, rep_conc_std = p.create_param_report(param='conc_vial_mg_L')
rep_sample_fr, rep_sample_fr_std = p.create_param_report(param='fraction_of_sample_fr')
# create aggreport (functionl group aggreageted based) for different parameters
agg_conc, agg_conc_std = p.create_param_aggrrep(param='conc_vial_mg_L')
agg_sample_fr, agg_sample_fr_std = p.create_param_aggrrep(param='fraction_of_sample_fr')
# plot results bases on report
#%%
p.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0,
    legend_location='outside', only_samples_to_plot=['A', 'Ader'], #y_lim=[0, 5000]
            )
# plot results bases on aggreport
p.plot_ave_std(param='fraction_of_sample_fr', aggr=True, min_y_thresh=0.01,
    y_lim=[0, .5], color_palette='Set2')