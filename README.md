# gcms_data_analysis

## A Python tool to manage multiple GCMS qualitative tables and automatically split chemicals into functional groups.

An open-source Python tool that can automatically:
- handle multiple GCMS semi-quantitative data tables (derivatized or not)
- duild a database of all identified compounds and their relevant properties using PubChemPy
- split each compound into its functional groups using a published fragmentation algorithm
- apply calibrations and/or semi-calibration using Tanimoto and molecular weight similarities
- produce single sample reports, comprehensive multi-sample reports and aggregated reports based on functional group mass fractions in the samples
- provides plotting capabilities

## Naming convention for samples

To ensure the code handles replicates of the same sample correctly, names have to follow the convention:
*name-of-sample-with-dashes-only_replicatenumber*

Examples that are *correctly* handled:
- Bio-oil-foodwaste-250C_1
- FW_2

Examples of *NON-ACCEPTABLE* names are
- bio_oil_1
- FW1

## Example

A comprehensive example is provided on the GitHub repository to show how inputs should be formatted.
To test the module, install the `gcms_data_analysis` module, download the example folder given in the repository, and run the example_gcms_data_analysis.py. The folder_path needs to be set to where your data folder is.

The example code is shown here for convenience:

<!-- EXAMPLE_START -->
```python

import pathlib as plib  # used for the folder_path
from gcms_data_analysis import Project

# you might need to change this to the path were you have the data
folder_path = plib.Path(plib.Path(__file__).cwd(), 'data')

# class methods need to be called at the beginning to influence all instances
Project.set_folder_path(folder_path)  # necessary for every project
Project.set_plot_grid(False)  # to make plots with gridlines
Project.set_plot_font('Sans')  # to use sans font in plots

# initialize project
p = Project()

# load files_info as provided by the user, if not given, create it
# using the GC-MS .txt files in the folder
files_info_0 = p.load_files_info()

# load the provided calibrations as dict, store bool to know if are deriv
calibrations, is_calibr_deriv = p.load_calibrations()
c1, c2 = calibrations['calibration'], calibrations['deriv_calibration']

# load provided classificaiton codes and mass fractions for fun. groups
class_code_frac = p.load_class_code_frac()

# load all GCMS txt files as single files
files0, is_files_deriv0 = p.load_all_files()
f1, f2, f3 = files0['A_1'], files0['Ader_1'], files0['B_1']

# create the list with all compounds in all samples
list_of_all_compounds = p.create_list_of_all_compounds()

# create the list with all derivatized compounds in all samples
list_of_all_deriv_compounds = p.create_list_of_all_deriv_compounds()

if 0: # set to 1 if you want to recreate compounds properties databases
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
rep_files_conc = p.create_files_param_report(param='conc_vial_mg_L')
rep_files_fr= p.create_files_param_report(param='fraction_of_sample_fr')
rep_samples_conc, rep_samples_conc_std = p.create_samples_param_report(param='conc_vial_mg_L')
rep_samples_fr, rep_samples_fr_std = p.create_samples_param_report(param='fraction_of_sample_fr')

# create aggreport (functionl group aggreageted based) for different parameters
agg_files_conc = p.create_files_param_aggrrep(param='conc_vial_mg_L')
agg_files_fr = p.create_files_param_aggrrep(param='fraction_of_sample_fr')
agg_samples_conc, agg_samples_conc_std = p.create_samples_param_aggrrep(param='conc_vial_mg_L')
agg_samples_fr, agg_samples_fr_std = p.create_samples_param_aggrrep(param='fraction_of_sample_fr')

# plot results bases on report
p.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0, files_or_samples='files',
    legend_location='outside',
    only_samples_to_plot=['A_1', 'A_2', 'Ader_1', 'Ader_2'], #y_lim=[0, 5000]
            )
# plot results bases on aggreport
p.plot_ave_std(param='fraction_of_sample_fr', aggr=True, files_or_samples='files',
                min_y_thresh=0.01,
    y_lim=[0, .5], color_palette='Set2')

p.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0,
    legend_location='outside', only_samples_to_plot=['A', 'Ader'], #y_lim=[0, 5000]
            )
# plot results bases on aggreport
p.plot_ave_std(param='fraction_of_sample_fr', aggr=True, min_y_thresh=0.01,
    y_lim=[0, .5], color_palette='Set2')

```
<!-- EXAMPLE_END -->