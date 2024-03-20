# Import necessary libraries
import pathlib as plib  # Used for handling file and directory paths
from gcms_data_analysis import (
    Project,
)  # Import the Project class from the gcms_data_analysis package

# Define the folder path where your data is located. Change this path to where you've stored your data files.
# folder_path = plib.Path(plib.Path(__file__).parent, "example\data")
folder_path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\example\data"
)

# Set global configurations for the Project class.
# These configurations affect all instances of the class.
Project.set_folder_path(
    folder_path
)  # Set the base folder path for the project's data files
Project.set_plot_grid(False)  # Disable grid lines in plots for a cleaner look
Project.set_plot_font("Sans")  # Set the font style for plots to 'Sans'

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
c1, c2 = calibrations["calibration"], calibrations["deriv_calibration"]

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
f11, f22, f33 = files["A_1"], files["Ader_1"], files["B_1"]

# Add statistical information to the files_info DataFrame, such as mean, median, and standard deviation for each file
files_info = gcms.add_stats_to_files_info()

# Create a samples_info DataFrame without applying calibration data, for initial analysis
samples_info_0 = gcms.create_samples_info()

# Create samples and their standard deviations from the files, storing the results in dictionaries
samples, samples_std = gcms.create_samples_from_files()
s1, s2, s3 = samples["A"], samples["Ader"], samples["B"]
sd1, sd2, sd3 = samples_std["A"], samples_std["Ader"], samples_std["B"]

# Generate reports for specific parameters (e.g., concentration, mass fraction) for files and samples
rep_files_conc = gcms.create_files_param_report(param="conc_vial_mg_L")
rep_files_fr = gcms.create_files_param_report(param="fraction_of_sample_fr")
rep_samples_conc, rep_samples_conc_std = gcms.create_samples_param_report(
    param="conc_vial_mg_L"
)
rep_samples_fr, rep_samples_fr_std = gcms.create_samples_param_report(
    param="fraction_of_sample_fr"
)

# Generate aggregated reports based on functional groups for files and samples, for specific parameters
agg_files_conc = gcms.create_files_param_aggrrep(param="conc_vial_mg_L")
agg_files_fr = gcms.create_files_param_aggrrep(param="fraction_of_sample_fr")
agg_samples_conc, agg_samples_conc_std = gcms.create_samples_param_aggrrep(
    param="conc_vial_mg_L"
)
agg_samples_fr, agg_samples_fr_std = gcms.create_samples_param_aggrrep(
    param="fraction_of_sample_fr"
)

# Plotting results based on the generated reports, allowing for visual comparison of average values and standard deviations
# Plot results for individual files or samples based

gcms.plot_ave_std(
    param="fraction_of_sample_fr",
    min_y_thresh=0,
    files_or_samples="files",
    legend_location="outside",
    only_samples_to_plot=["A_1", "A_2", "Ader_1", "Ader_2"],  # y_lim=[0, 5000]
)
# plot results bases on aggreport
gcms.plot_ave_std(
    param="fraction_of_sample_fr",
    aggr=True,
    files_or_samples="files",
    min_y_thresh=0.01,
    y_lim=[0, 0.5],
    color_palette="Set2",
)

gcms.plot_ave_std(
    param="fraction_of_sample_fr",
    min_y_thresh=0,
    legend_location="outside",
    only_samples_to_plot=["A", "Ader"],  # y_lim=[0, 5000]
)
# plot results bases on aggreport
gcms.plot_ave_std(
    param="fraction_of_sample_fr",
    aggr=True,
    min_y_thresh=0.01,
    y_lim=[0, 0.5],
    color_palette="Set2",
)
