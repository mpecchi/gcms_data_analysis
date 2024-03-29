# %% Import necessary libraries
import pathlib as plib  # Used for handling file and directory paths
from gcms_data_analysis import Project
from gcms_data_analysis.plotting import plot_ave_std

# Define the folder path where your data is located. Change this path to where you've stored your data files.
# folder_path = plib.Path(plib.Path(__file__).parent, "example\data")
folder_path = plib.Path(
    r"path\to\data\folder\"
)
# folder_path: plib.Path = plib.Path(
#     r"C:\Users\mp933\OneDrive - Cornell University\Python\GCMS\NNDNDD"
# )
# Set global configurations for the Project class.
# These configurations affect all instances of the class.
Project.set_folder_path(
    folder_path
)  # Set the base folder path for the project's data files
Project.set_plot_grid(False)  # Disable grid lines in plots for a cleaner look
Project.set_plot_font("Sans")  # Set the font style for plots to 'Sans'
Project.set_auto_save_to_excel(False)
# Initialize a Project instance to manage and analyze GCMS data
gcms = Project()

# Load metadata from a user-provided 'files_info.xlsx' file, or generate it from .txt GC-MS files if not provided
files_info = gcms.load_files_info()
# Load individual GCMS .txt files as pandas DataFrames
files = gcms.load_all_files()
files = gcms.add_iupac_to_files()
list_of_all_compounds = gcms.create_list_of_all_compounds()
files, is_files_deriv = gcms.apply_calibration_to_files()
samples_info, samples_info_std = gcms.create_samples_info()
samples, samples_std = gcms.create_samples_from_files()

params = [
    "height",
    "area",
    "area_if_undiluted",
    "conc_vial_mg_L",
    "conc_vial_if_undiluted_mg_L",
    "fraction_of_sample_fr",
    "fraction_of_feedstock_fr",
]
for param in params:
    _ = gcms.create_files_param_report(param)
    _ = gcms.create_files_param_aggrrep(param)

    _, _ = gcms.create_samples_param_report(param)
    _, _ = gcms.create_samples_param_aggrrep(param)


# Plotting results based on the generated reports, allowing for visual comparison of average values and standard deviations
# Plot results for individual files or samples based

plot_ave_std(
    gcms,
    param="fraction_of_sample_fr",
    min_y_thresh=0,
    files_or_samples="files",
    legend_location="outside",
    only_samples_to_plot=["S_1", "S_2", "T_1", "T_2"],  # y_lim=[0, 5000]
)
# plot results bases on aggreport
plot_ave_std(
    gcms,
    param="fraction_of_sample_fr",
    aggr=True,
    files_or_samples="files",
    min_y_thresh=0.01,
    y_lim=[0, 0.5],
    color_palette="Set2",
)

plot_ave_std(
    gcms,
    param="fraction_of_sample_fr",
    min_y_thresh=0,
    legend_location="outside",
    only_samples_to_plot=["S", "T"],  # y_lim=[0, 5000]
)
# plot results bases on aggreport
plot_ave_std(
    gcms,
    param="fraction_of_sample_fr",
    aggr=True,
    min_y_thresh=0.01,
    y_lim=[0, 0.5],
    color_palette="Set2",
)

# %%
# import pickle

# folder_path: plib.Path = plib.Path(r"C:\Users\mp933\Desktop\New folder")
# pickle_path: plib.Path = plib.Path(folder_path, "pickle_object.pkl")
# with open(pickle_path, "wb") as output_file:
#     pickle.dump(gcms, output_file)
# %%
# import pickle
# import pathlib as plib  # Used for handling file and directory paths
# from gcms_data_analysis import (
#     Project,
# )  # Import the Project class from the gcms_data_analysis package

# folder_path: plib.Path = plib.Path(r"C:\Users\mp933\Desktop\New folder")
# pickle_path: plib.Path = plib.Path(folder_path, "pickle_object.pkl")
# with open(pickle_path, "rb") as input_file:
#     gcms: Project = pickle.load(input_file)
# from gcms_data_analysis.plotting import plot_pave_std

# # %%
# myfig = plot_pave_std(
#     gcms,
#     files_or_samples="files",
#     width=12,
#     height=5,
#     legend_location="outside",
#     y_lim=[0, 100],
# )
# # %%
# myfig = plot_pave_std(
#     gcms,
#     files_or_samples="samples",
#     width=6,
#     height=6,
#     legend_location="best",
#     y_lim=[0, 100],
#     min_y_thresh=10,
# )

# # %%
