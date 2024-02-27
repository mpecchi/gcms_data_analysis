"""
REFER TO THE WORD README ASSOCIATED

GC-MS data handler (READ THIS FULLY BEFORE USING THE CODE)
@author Matteo Pecchi (mp933@cornell.edu).

"""
# =============================================================================
# # necessary packages, install them using conda (not pip)
# =============================================================================
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import pathlib as plib
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from gcms_data_analysis import Project, figure_create, figure_save

folder_path = plib.Path(plib.Path(__file__).parent, 'data')
#%%
def get_calibration_error(name0, name1, calibration, xrange=[10, 200], steps=100):
    cols_cal_area = [c for c in list(calibration) if 'Area' in c]
    cols_cal_ppms = [c for c in list(calibration) if 'PPM' in c]
    calibration[cols_cal_area + cols_cal_ppms] = \
    calibration[cols_cal_area + cols_cal_ppms].apply(pd.to_numeric,
                                               errors='coerce')
    cal_areas0 = calibration.loc[name0, cols_cal_area].to_numpy(dtype=float)
    cal_ppms0 = calibration.loc[name0, cols_cal_ppms].to_numpy(dtype=float)
    # linear fit of calibration curve (exclude nan), get ppm from area
    fit0 = np.polyfit(cal_areas0[~np.isnan(cal_areas0)],
                      cal_ppms0[~np.isnan(cal_ppms0)], 1)
    cal_areas1 = calibration.loc[name1, cols_cal_area].to_numpy(dtype=float)
    cal_ppms1 = calibration.loc[name1, cols_cal_ppms].to_numpy(dtype=float)
    # linear fit of calibration curve (exclude nan), get ppm from area
    fit1 = np.polyfit(cal_areas1[~np.isnan(cal_areas1)],
                      cal_ppms1[~np.isnan(cal_ppms1)], 1)
    x = np.arange(xrange[0], xrange[1], steps)
    line0 = np.poly1d(fit0)(x)
    line1 = np.poly1d(fit1)(x)

    mse = np.mean((line0 - line1)**2)
    mse = np.average(abs(line0-line1)/line1)*100
    return mse

#%%

Project.set_folder_path(folder_path)
# Set the base folder path for the project's data files
Project.set_folder_path(folder_path)

# Initialize a Project instance to manage and analyze GCMS data
gcms = Project()

# Load metadata from a user-provided 'files_info.xlsx' file, or generate it from .txt GC-MS files if not provided
files_info0 = gcms.load_files_info()

# Load individual GCMS .txt files as pandas DataFrames
files = gcms.load_all_files()

# Load classification codes and mass fractions for functional groups from a provided file
class_code_frac = gcms.load_class_code_frac()

# Load calibration data for standard and derivatized samples, and determine if they are derivatized
calibrations, is_calibr_deriv = gcms.load_calibrations()
c1, c2 = calibrations['calibration88'], calibrations['deriv_calibration11']

# Generate a comprehensive list of all compounds found across samples
list_of_all_compounds = gcms.create_list_of_all_compounds()

# Similarly, create a list of all derivatized compounds found across samples
list_of_all_deriv_compounds = gcms.create_list_of_all_deriv_compounds()

# Load properties for standard and derivatized compounds from provided files
compounds_properties = gcms.load_compounds_properties()
deriv_compounds_properties = gcms.load_deriv_compounds_properties()

# Flag indicating whether new compounds have been added, triggering a need to regenerate properties data
new_files_with_new_compounds_added = False
if new_files_with_new_compounds_added:
    compounds_properties = gcms.create_compounds_properties()
    deriv_compounds_properties = gcms.create_deriv_compounds_properties()

# Apply calibration data to all loaded files, adjusting compound concentrations based on calibration curves
files, is_files_deriv = gcms.apply_calibration_to_files()

# Extract specific files for detailed analysis or further operations
f11, f22, f33 = files['A_1'], files['Ader_1'], files['B_1']

# Add statistical information to the files_info DataFrame, such as mean, median, and standard deviation for each file
files_info = gcms.add_stats_to_files_info()

# Create a samples_info DataFrame without applying calibration data, for initial analysis
samples_info_0 = gcms.create_samples_info()

# Create samples and their standard deviations from the files, storing the results in dictionaries
samples, samples_std = gcms.create_samples_from_files()
s1, s2, s3 = samples['A'], samples['Ader'], samples['B']
sd1, sd2, sd3 = samples_std['A'], samples_std['Ader'], samples_std['B']

# Add statistical information to the samples_info DataFrame, enhancing the initial analysis with statistical data
samples_info = gcms.add_stats_to_samples_info()

# Generate reports for specific parameters (e.g., concentration, mass fraction) for files and samples
rep_files_conc = gcms.create_files_param_report(param='conc_vial_mg_L')
rep_files_fr = gcms.create_files_param_report(param='fraction_of_sample_fr')
rep_samples_conc, rep_samples_conc_std = gcms.create_samples_param_report(param='conc_vial_mg_L')
rep_samples_fr, rep_samples_fr_std = gcms.create_samples_param_report(param='fraction_of_sample_fr')

# Generate aggregated reports based on functional groups for files and samples, for specific parameters
agg_files_conc = gcms.create_files_param_aggrrep(param='conc_vial_mg_L')
agg_files_fr = gcms.create_files_param_aggrrep(param='fraction_of_sample_fr')
agg_samples_conc, agg_samples_conc_std = gcms.create_samples_param_aggrrep(param='conc_vial_mg_L')
agg_samples_fr, agg_samples_fr_std = gcms.create_samples_param_aggrrep(param='fraction_of_sample_fr')

# %% Plotting results based on the generated reports, allowing for visual comparison of average values and standard deviations
# Plot results for individual files or samples based

gcms.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0.05, files_or_samples='files',
    legend_location='outside', xlab_rot=30, filename='sample_fraction_files',
    # only_samples_to_plot=['A_1', 'A_2', 'Ader_1', 'B_2'],
    y_lim=[0, .3], annotate_lttrs='a'
            )
gcms.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0.05, files_or_samples='samples',
    legend_location='outside', xlab_rot=0, filename='sample_fraction_samples',
    # only_samples_to_plot=['A_1', 'A_2', 'Ader_1', 'B_2'],
    y_lim=[0, .3], annotate_lttrs='b'
            )
#%%
# plot results bases on aggreport
gcms.plot_ave_std(param='fraction_of_sample_fr', aggr=True, files_or_samples='files',
    filename='sample_fraction_aggr_files', xlab_rot=30, annotate_lttrs='c',
    min_y_thresh=0.01, #yt_sum=True,
    y_lim=[0, 1], color_palette='Set2')
gcms.plot_ave_std(param='fraction_of_sample_fr', aggr=True, files_or_samples='samples',
    filename='sample_fraction_aggr_samples', annotate_lttrs='d',
    min_y_thresh=0.01, #yt_sum=True,
    y_lim=[0, 1], color_palette='Set2')
#%%
gcms.plot_ave_std(param='fraction_of_sample_fr', min_y_thresh=0.01,
    legend_location='outside', only_samples_to_plot=['A', 'Ader', 'B'],
    y_lim=[0, 0.3]
            )
# %% plot results bases on aggreport
gcms.plot_ave_std(param='fraction_of_sample_fr', aggr=True, min_y_thresh=0.01,
    y_lim=[0, .5], color_palette='Set2')

#%%
run_tanimoto_analysis = True
if run_tanimoto_analysis:
    in_path = folder_path
    out_path_cal = plib.Path(folder_path, 'output_tanimoto')
    out_path_cal.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_excel(plib.Path(in_path, 'calibration88.xlsx'),
                        engine='openpyxl', index_col='Name')

    combs = combinations(calibration.index.tolist(), 2)
    tanimoto_error = pd.DataFrame(columns=['CalErr', 'DistMW', 'TanimS'], index=range(3915))
    for c, (name0, name1) in enumerate(combs):
        tanimoto_error.loc[c, 'CalErr'] = get_calibration_error(name0, name1, calibration)
        tanimoto_error.loc[c, 'DistMW'] = abs(calibration.loc[name0,'MW'] - calibration.loc[name1,'MW'])
        try:
            smis = [calibration.loc[name0, 'canonical_smiles'],
                    calibration.loc[name1, 'canonical_smiles']]
            mols = [Chem.MolFromSmiles(smi) for smi in smis]
            fps = [GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
            # perform Tanimoto similarity
            tanimoto_error.loc[c, 'TanimS'] = DataStructs.TanimotoSimilarity(fps[0], fps[1])
        except TypeError:
            tanimoto_error.loc[c, 'TanimS'] = np.nan
    tanimoto_error.to_excel(plib.Path(out_path_cal, 'tanimoto_error.xlsx'))
    fig, ax, axt, fig_par = figure_create(rows=1, cols=1, plot_type=0, hgt_mltp=1.2,
                                    paper_col=1.4)

    aa = ax[0].scatter(tanimoto_error['TanimS'].values, tanimoto_error['CalErr'].values,
                    c=tanimoto_error['DistMW'].values)
    ax[0].set_yscale('log')
    plt.colorbar(aa, label=r'$\Delta$MW [atomic mass unit]')
    plt.hlines(y=100, xmin=0, xmax=1, color='grey', linestyle='dotted')
    plt.vlines(x=.4, ymin=0, ymax=100, color='grey', linestyle='dashed')
    ax[0].annotate('default\nsetting', ha='left', va='bottom',
                xycoords='axes fraction',
                xy=(0.3, .01))
    ax[0].annotate('Error = 100%', ha='left', va='bottom',
                xycoords='axes fraction',
                xy=(0.8, .6))
    figure_save('tanimoto_error', out_path_cal, fig, ax, axt, fig_par,
            x_lab='Tanimoto Similarity [-]', x_lim=[0, 1], y_lab='Average error [%]',
            legend=None, tight_layout=True)

    # create and export the similarity table for tetradecanoic acid
    cpmnds = gcms.compounds_properties.set_index('iupac_name')
    cpmnds = cpmnds[~cpmnds.index.duplicated(keep='first')].copy()
    iupac = cpmnds.index[0]
    mws = [cpmnds.loc[iupac, 'molecular_weight']]
    smis = [cpmnds.loc[iupac, 'canonical_smiles']]
    names_cal = [iupac]
    # then add all properties for all calibrated compounds
    # if the sample was not derivatized (default)
    # if not self.is_files_deriv[filename]:
    for c in cpmnds.index.tolist()[1:6]:
        names_cal.append(c)
        # print(df_comps.index)
        smis.append(cpmnds.loc[c, 'canonical_smiles'])
        mws.append(cpmnds.loc[c, 'molecular_weight'])
        # calculate the delta mw with all calib compounds
        delta_mw = np.abs(np.asarray(mws)[0]
                    - np.asarray(mws)[1:])
        # get mols and fingerprints from rdkit for each comp
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        fps = [GetMorganFingerprintAsBitVect(ml, 2, nBits=1024)
        for ml in mols]
        # perform Tanimoto similarity betwenn the first and all
        # other compounds
        s = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
        # create a df with results
        df_sim = pd.DataFrame(data={'name': names_cal[1:],
        'smiles': smis[1:], 'Similarity': s, 'delta_mw': delta_mw})
        # put the index title as the comp
        df_sim.set_index('name', inplace=True)
        df_sim.index.name = iupac
        # sort values based on similarity and delta mw
        df_sim = df_sim.sort_values(['Similarity', 'delta_mw'],
                                ascending=[False, True])
    df_sim.to_excel(plib.Path(out_path_cal, 'similarity_table_tetradecanoic.xlsx'))

# %%

