# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:45:31 2023

@author: mp933
"""

import pathlib as plib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import ele
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdmolops
from rdkit.Chem.AllChem import (
    GetMorganFingerprintAsBitVect,
)  # pylint: disable=no-name-in-module
from gcms_data_analysis.fragmenter import Fragmenter


def get_compound_from_pubchempy(comp_name: str) -> pcp.Compound:
    if not isinstance(comp_name, str) or comp_name.isspace():
        print(f"WARNING get_compound_from_pubchempy got an invalid {comp_name =}")
        return None
    cond = True
    while cond:  # to deal with HTML issues on server sides (timeouts)
        try:
            # comp contains all info about the chemical from pubchem
            try:
                comp_inside_list = pcp.get_compounds(comp_name, "name")
            except ValueError:
                print(f"{comp_name = }")
                return None
            if comp_inside_list:
                comp = comp_inside_list[0]
            else:
                print(
                    f"WARNING: name_to_properties {comp_name=} does not find an entry in pcp",
                )
                return None
            cond = False
        except pcp.PubChemHTTPError:  # timeout error, simply try again
            print("Caught: pcp.PubChemHTTPError (keep trying)")
    return comp


def _order_columns_in_compounds_properties(
    unsorted_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if unsorted_df is None:
        return None
    priority_cols: list[str] = [
        "iupac_name",
        "underiv_comp_name",
        "molecular_formula",
        "canonical_smiles",
        "molecular_weight",
        "xlogp",
    ]

    # Define a custom sort key function
    def sort_key(col):
        if col in priority_cols:
            return (-1, priority_cols.index(col))
        if col.startswith("el_mf"):
            return (2, col)
        elif col.startswith("el_"):
            return (1, col)
        elif col.startswith("fg_mf_unclassified"):
            return (5, col)
        elif col.startswith("fg_mf"):
            return (4, col)
        elif col.startswith("fg_"):
            return (3, col)
        else:
            return (0, col)

    # Sort columns using the custom key
    sorted_columns = sorted(unsorted_df.columns, key=sort_key)
    sorted_df = unsorted_df.reindex(sorted_columns, axis=1)
    sorted_df.index.name = "comp_name"
    # Reindex the DataFrame with the sorted columns
    return sorted_df


def name_to_properties(
    comp_name: str,
    dict_classes_to_codes: dict[str:str],
    dict_classes_to_mass_fractions: dict[str:float],
    df: pd.DataFrame = pd.DataFrame(),
    precision_sum_elements: float = 0.05,
    precision_sum_functional_group: float = 0.05,
) -> pd.DataFrame:
    """
    used to retrieve chemical properties of the compound indicated by the
    comp_name and to store those properties in the df

    Parameters
    ----------
    GCname : str
        name from GC, used as a unique key.
    search_name : str
        name to be used to search on pubchem.
    df : pd.DataFrame
        that contains all searched compounds.
    df_class_code_frac : pd.DataFrame
        contains the list of functional group names, codes to be searched
        and the weight fraction of each one to automatically calculate the
        mass fraction of each compounds for each functional group.
        Classes are given as smarts and are looked into the smiles of the comp.

    Returns
    -------
    df : pd.DataFrame
        updated dataframe with the searched compound.
    CompNotFound : str
        if GCname did not yield anything CompNotFound=GCname.

    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The argument df must be a pd.DataFrame.")

    if not isinstance(comp_name, str) or comp_name.isspace():
        return _order_columns_in_compounds_properties(df)

    if comp_name in df.index.tolist():
        return _order_columns_in_compounds_properties(df)

    comp = get_compound_from_pubchempy(comp_name)

    if comp is None:
        df.loc[comp_name, "iupac_name"] = "unidentified"
        return _order_columns_in_compounds_properties(df)

    try:
        valid_iupac_name = comp.iupac_name.lower()
    except AttributeError:  # iupac_name not give
        valid_iupac_name = comp_name.lower()

    df.loc[comp_name, "iupac_name"] = valid_iupac_name
    df.loc[comp_name, "molecular_formula"] = comp.molecular_formula
    df.loc[comp_name, "canonical_smiles"] = comp.canonical_smiles
    df.loc[comp_name, "molecular_weight"] = float(comp.molecular_weight)

    try:
        df.loc[comp_name, "xlogp"] = float(comp.xlogp)
    except (
        TypeError
    ):  # float() argument must be a string or a real number, not 'NoneType'
        df.loc[comp_name, "xlogp"] = np.nan
    elements = set(comp.to_dict()["elements"])
    el_dict = {}
    el_mf_dict = {}

    for el in elements:
        el_count = comp.to_dict()["elements"].count(el)
        el_mass = ele.element_from_symbol(el).mass

        # Using similar logic as in the fg_dict example
        if el not in el_dict:
            el_dict[el] = 0
            el_mf_dict[el] = 0.0

        el_dict[el] += int(el_count)
        el_mf_dict[el] += (
            float(el_count) * float(el_mass) / float(comp.molecular_weight)
        )
    # Now, update the DataFrame in a similar way to the fg_dict example
    for key, value in el_dict.items():
        df.at[comp_name, f"el_{key}"] = int(value)

    for key, value in el_mf_dict.items():
        df.at[comp_name, f"el_mf_{key}"] = float(value)
    cols_el_mf = [col for col in df.columns if col.startswith("el_mf_")]
    residual_els = df.loc[comp_name, cols_el_mf].sum() - 1
    # check element sum
    try:
        assert residual_els <= precision_sum_elements
    except AssertionError:
        print(f"the total mass fraction of elements in {comp_name =} is > 0.001")
    # apply fragmentation using the Fragmenter class (thanks simonmb)
    frg = Fragmenter(
        dict_classes_to_codes,
        fragmentation_scheme_order=dict_classes_to_codes.keys(),
        algorithm="simple",
    )
    fragmentation, _, _ = frg.fragment(comp.canonical_smiles)
    fg_dict = {}
    fg_mf_dict = {}
    # Iterate over each item in the dictionary
    for key, value in fragmentation.items():
        # Determine the root key (the part before an underscore, if present)
        root_key = key.split("_")[0]
        # if root_key in hetero_atoms:
        #     pass
        # Check if the root key is in the sum_dict; if not, initialize it
        if root_key not in fg_dict:
            fg_dict[root_key] = 0
            fg_mf_dict[root_key] = 0
        # Add the value to the corresponding root key in the sum_dict
        fg_dict[root_key] += int(fragmentation[key])
        fg_mf_dict[root_key] += (
            float(fragmentation[key])
            * float(dict_classes_to_mass_fractions[key])
            / df.loc[comp_name, "molecular_weight"].astype(float)
        )  # mass fraction of total

    # Update df with fg_dict
    for key, value in fg_dict.items():
        df.at[comp_name, f"fg_{key}"] = int(value)  # Update the cell
    # Update df with fg_mf_dict
    for key, value in fg_mf_dict.items():
        df.at[comp_name, f"fg_mf_{key}"] = float(value)  # Update the cell
    cols_fg_mf = [col for col in df.columns if col.startswith("fg_mf")]
    residual_fgs = df.loc[comp_name, cols_fg_mf].sum() - 1
    try:
        assert residual_fgs <= precision_sum_functional_group
    except AssertionError:
        print(f"{df.loc[comp_name, cols_fg_mf].sum()=}")
        print(
            f"the total mass fraction of functional groups in {comp_name =} is > 0.05"
        )
    if residual_fgs < -precision_sum_functional_group:
        df.at[comp_name, "fg_mf_unclassified"] = abs(residual_fgs)
    df.loc[df["iupac_name"] != "unidentified"] = df.loc[
        df["iupac_name"] != "unidentified"
    ].fillna(0)
    return _order_columns_in_compounds_properties(df)


# %%
def get_iupac_from_pcp(comp_name: str) -> str:
    """get iupac name for compound using pubchempy, needs internet connection

    :param comp_name: _description_
    :type comp_name: str
    :return: lowercase iupac name for the compound
    :rtype: str
    """
    cond = True
    while cond:  # to deal with HTML issues on server sides (timeouts)
        try:
            # comp contains all info about the chemical from pubchem
            try:
                comp = pcp.get_compounds(comp_name, "name")[0]

            except ValueError:
                print(f"Calibration iupac addition: compund {comp_name} did not work")
            try:
                iup: str = comp.iupac_name
            except AttributeError:  # iupac_name not give
                print(
                    f"Calibration iupac addition: compund {comp_name} does not have a iupac entry"
                )
            cond = False
        except pcp.PubChemHTTPError:  # timeout error, simply try again
            print("Caught: pcp.PubChemHTTPError")
    return iup.lower()


def report_difference(rep1, rep2, diff_type="absolute"):
    """
    calculates the ave, std and p percentage of the differnece between
    two reports where columns and index are the same.
    Replicates (indicated as XX_1, XX_2) are used for std.

    Parameters
    ----------
    rep1 : pd.DataFrame
        report that is conisdered the reference to compute differences from.
    rep2 : pd.DataFrame
        report with the data to compute the difference.
    diff_type : str, optional
        type of difference, absolute vs relative (to rep1)
        . The default is 'absolute'.

    Returns
    -------
    dif_ave : pd.DataFrame
        contains the average difference.
    dif_std : pd.DataFrame
        contains the std, same units as dif_ave.
    dif_stdp : pd.DataFrame
        contains the percentage std compared to ref1.

    """
    idx_name = rep1.index.name
    rep1 = rep1.transpose()
    rep2 = rep2.transpose()

    # put the exact same name on files (by removing the '_#' at end)
    repl_idx1 = [i if "_" not in i else i.split("_")[0] for i in rep1.index.tolist()]
    repl_idx2 = [i if "_" not in i else i.split("_")[0] for i in rep2.index.tolist()]
    rep1.loc[:, idx_name] = repl_idx1
    rep2.loc[:, idx_name] = repl_idx2
    # compute files and std of files and update the index
    rep_ave1 = rep1.groupby(idx_name, sort=False).mean().reset_index()
    rep_std1 = rep1.groupby(idx_name, sort=False).std().reset_index()
    rep_ave1.set_index(idx_name, inplace=True)
    rep_std1.set_index(idx_name, inplace=True)
    rep_ave2 = rep2.groupby(idx_name, sort=False).mean().reset_index()
    rep_std2 = rep2.groupby(idx_name, sort=False).std().reset_index()
    rep_ave2.set_index(idx_name, inplace=True)
    rep_std2.set_index(idx_name, inplace=True)

    if diff_type == "absolute":
        dif_ave = rep_ave1 - rep_ave2
        dif_std = np.sqrt(rep_std1**2 + rep_std2**2)
        dif_stdp = np.sqrt(rep_std1**2 + rep_std2**2) / dif_ave * 100
    if diff_type == "relative":
        dif_ave = (rep_ave1 - rep_ave2) / rep_ave1
        dif_std = np.sqrt(rep_std1**2 + rep_std2**2) / rep_ave1
        dif_stdp = np.sqrt(rep_std1**2 + rep_std2**2) / rep_ave1 / dif_ave * 100

    return dif_ave, dif_std, dif_stdp


class Project:
    """the class that contains all method and info to analyze
    the project (intended as a collection of GCMS files, calibrations, etc)
    """

    folder_path = plib.Path.cwd()
    in_path = folder_path
    out_path = plib.Path(in_path, "output")
    shared_path = in_path.parents[0]
    auto_save_to_excel = True
    plot_font = "Dejavu Sans"
    plot_grid = False
    load_delimiter = "\t"
    load_skiprows = 8
    columns_to_keep_in_files = ["Ret.Time", "Height", "Area", "Name"]
    columns_to_rename_in_files = {
        "Ret.Time": "retention_time",
        "Height": "height",
        "Area": "area",
        "Name": "comp_name",
    }

    compounds_to_rename = {}
    param_to_axis_label: dict[str:str] = {
        "height": "Peak Height [-]",
        "area": "Peak Area [-]",
        "area_if_undiluted": "Peak Area [-]",
        "conc_vial_mg_L": "conc. [mg/L] (ppm)",
        "conc_vial_if_undiluted_mg_L": "conc. [mg/L] (ppm)",
        "fraction_of_sample_fr": "mass fraction [g/g$_{sample}$]",
        "fraction_of_feedstock_fr": "mass fraction [g/g$_{feedstock}$]",
    }
    acceptable_params: list[str] = list(param_to_axis_label.keys())
    string_in_deriv_names: list[str] = [
        "deriv",
        "tms",
        "tbms",
        "trimethylsilyl",
    ]
    string_in_deriv_names = [s.lower() for s in string_in_deriv_names]
    files_info_defauls_columns = [
        "dilution_factor",
        "total_sample_conc_in_vial_mg_L",
        "sample_yield_on_feedstock_basis_fr",
    ]
    semi_calibration = True
    tanimoto_similarity_threshold = 0.4
    delta_mol_weight_threshold = 100
    column_to_sort_values_in_samples = "retention_time"

    @classmethod
    def set_folder_path(cls, path):
        """Set the folder path for the project. This method updates the
        class attributes related to the project's directory structure,
        including input, output (default 'output'), and shared paths. The default
        folder path is the current working directory."""
        cls.folder_path = plib.Path(path).resolve()
        cls.in_path = cls.folder_path
        cls.out_path = plib.Path(cls.in_path, "output")
        plib.Path(cls.out_path).mkdir(parents=True, exist_ok=True)
        cls.shared_path = cls.in_path.parents[0]

    @classmethod
    def set_auto_save_to_excel(cls, new_auto_save_to_excel: bool):
        """Enable or disable automatic saving of results to Excel (default True).
        This method updates the class attribute that controls whether
        analysis results are automatically saved in an Excel file."""
        cls.auto_save_to_excel = new_auto_save_to_excel

    @classmethod
    def set_plot_font(cls, new_plot_font):
        """Set the font used in plots (default 'Dejavu Sans'). This method updates
        the class attribute that specifies the font style used in graphical plots
        generated by the project."""
        cls.plot_font = new_plot_font

    @classmethod
    def set_tanimoto_similarity_threshold(cls, new_tanimoto_similarity_threshold):
        """Set the Tanimoto similarity threshold for compound matching (default 0.4).
        This method updates the class attribute that specifies the threshold used to
        determine compound similarity based on Tanimoto score."""
        cls.tanimoto_similarity_threshold = new_tanimoto_similarity_threshold

    @classmethod
    def set_delta_mol_weight_threshold(cls, new_delta_mol_weight_threshold):
        """Set the delta molecular weight threshold for compound matching (default 100).
        This method updates the class attribute that specifies the threshold used for
        comparing molecular weights in compound matching."""
        cls.delta_mol_weight_threshold = new_delta_mol_weight_threshold

    @classmethod
    def set_plot_grid(cls, new_plot_grid):
        """Enable or disable grid in plots (default False). This method updates the class
        attribute that controls the visibility of the grid in graphical plots generated by
        the project."""
        cls.plot_grid = new_plot_grid

    @classmethod
    def set_load_skiprows(cls, new_load_skiprows):
        """Set the number of rows to skip when loading data files (default 8). This method
        updates the class attribute that specifies how many initial rows should be skipped
        during the data loading process."""
        cls.load_skiprows = new_load_skiprows

    @classmethod
    def set_load_delimiter(cls, new_load_delimiter):
        """Set the delimiter used for loading data files (default '\t'). This method updates
        the class attribute that specifies the delimiter character used in the data files
        to be loaded."""
        cls.load_delimiter = new_load_delimiter

    @classmethod
    def set_columns_to_keep_in_files(cls, new_columns_to_keep_in_files):
        """Update the list of columns to retain from data files (default includes ['Ret.Time',
        'Height', 'Area', 'Name']). This method updates the class attribute that specifies
        which columns should be kept during the data processing."""
        cls.columns_to_keep_in_files = new_columns_to_keep_in_files

    @classmethod
    def set_string_in_deriv_names(cls, new_string_in_deriv_names):
        """Update the strings identifying derivatized compounds (default includes ['deriv.',
        'derivative', 'TMS', 'TBDMS', 'trimethylsilyl']). This method updates the class
        attribute with a list of strings used to identify derivatized compounds in the data.
        """
        cls.string_in_deriv_names = new_string_in_deriv_names

    @classmethod
    def set_compounds_to_rename(cls, new_compounds_to_rename):
        """Update the mapping of compounds to new names. This method updates the class
        attribute that holds a dictionary mapping original compound names to their new
        names as specified by the user. There is no default mapping."""
        cls.compounds_to_rename = new_compounds_to_rename

    @classmethod
    def set_param_to_axis_label(cls, new_param_to_axis_label):
        """Update the mapping of analysis parameters to axis labels for plots. This method
        updates the class attribute that holds a dictionary mapping analysis parameters to
        their corresponding axis labels in plots. Default mappings include 'area' to 'Peak
        Area [-]', 'conc_vial_mg_L' to 'conc. [mg/L] (ppm)', etc."""
        cls.param_to_axis_label = new_param_to_axis_label

    @classmethod
    def set_column_to_sort_values_in_samples(cls, new_column_to_sort_values_in_samples):
        """Update the column that is used to sort the entries (compounds) in each sample.
        Default is retention_time, alternative is area"""
        cls.column_to_sort_values_in_samples = new_column_to_sort_values_in_samples

    def __init__(self):
        """ """
        self.files_info: pd.DataFrame | None = None
        self.files_info_created: bool = False
        self.deriv_files_present: bool = False
        self.class_code_frac: pd.DataFrame | None = None
        self.class_code_frac_loaded: bool = False
        self.calibrations: dict[str : pd.DataFrame] = {}
        self.is_calibrations_deriv: dict[str:bool] = {}
        self.calibrations_loaded: bool = False
        self.calibrations_not_present: bool = False
        self.list_of_all_compounds: list[str] = []
        self.list_of_all_deriv_compounds: list[str] = []
        self.list_of_all_compounds_created = False
        self.list_of_all_deriv_compounds_created = False
        self.compounds_properties = None
        self.deriv_compounds_properties = None
        self.compounds_properties_created = False
        self.deriv_compounds_properties_created = False
        self.files_info = None
        self.files = {}
        self.is_files_deriv = {}
        self.files_loaded = False
        self.iupac_to_files_added = False
        self.iupac_to_calibrations_added = False
        self.calibration_to_files_applied = False
        self.stats_to_files_info_added = False

        self.samples_info = None
        self.samples_info_std = None
        self.samples_info_created = False
        self.stats_to_samples_info_added = False
        self.samples = {}
        self.samples_std = {}
        self.samples_created = False

        self.list_of_files_param_reports = []
        self.list_of_files_param_aggrreps = []
        self.list_of_samples_param_reports = []
        self.list_of_samples_param_aggrreps = []

        self.files_reports = {}
        self.files_aggrreps = {}
        self.samples_reports = {}
        self.samples_reports_std = {}
        self.samples_aggrreps = {}
        self.samples_aggrreps_std = {}

        # self.load_files_info()

    def load_files_info(self):
        """Attempts to load the 'files_info.xlsx' file containing metadata about GCMS
        files. If the file is not found, it creates a new 'files_info' DataFrame with
        default values based on the GCMS files present in the project's input path and
        saves it to 'files_info.xlsx'. This method ensures 'files_info' is loaded with
        necessary defaults and updates the class attribute 'files_info_created' to True.
        """
        try:
            files_info_no_defaults = pd.read_excel(
                plib.Path(Project.in_path, "files_info.xlsx"),
                engine="openpyxl",
                index_col="filename",
            )
            files_info = self._add_default_to_files_info(files_info_no_defaults)
            print("Info: files_info loaded")
            if Project.auto_save_to_excel:
                files_info.to_excel(plib.Path(Project.out_path, "files_info.xlsx"))
            self.files_info = files_info
            self.files_info_created = True
            if any(files_info["derivatized"]):
                self.deriv_files_present = True
                print("Info: derivatized samples are present")
        except FileNotFoundError:
            print("Info: files_info not found")
            files_info = self.create_files_info()
        return files_info

    def create_files_info(self):
        """Creates a default 'files_info' DataFrame from GCMS files found in the project's
        input path if an existing 'files_info' file is not found. It autogenerates filenames,
        samples, and replicates based on the GCMS file names, saves the DataFrame to
        'files_info.xlsx', and sets it as the current 'files_info' attribute."""
        filename = sorted(
            [a.parts[-1].split(".")[0] for a in list(Project.in_path.glob("**/*.txt"))]
        )
        samplename = [f.split("_")[0] for f in filename]
        replicate_number = [f.split("_")[1] for f in filename]
        files_info_no_defaults = pd.DataFrame(
            {
                "filename": filename,
                "samplename": samplename,
                "replicate_number": replicate_number,
            }
        )
        files_info_no_defaults.set_index("filename", drop=True, inplace=True)
        files_info = self._add_default_to_files_info(files_info_no_defaults)
        print("Info: files_info created")
        if Project.auto_save_to_excel:
            files_info.to_excel(plib.Path(Project.out_path, "files_info.xlsx"))
        self.files_info = files_info
        self.files_info_created = True
        if any(files_info["derivatized"]):
            self.deriv_files_present = True
            print("Info: derivatized samples are present")
        return files_info

    def _add_default_to_files_info(self, files_info_no_defaults):
        """Adds default values to the 'files_info' DataFrame for missing essential columns.
        This method ensures that every necessary column exists in 'files_info', filling
        missing ones with default values or false flags, applicable for both user-provided
        and automatically created 'files_info' DataFrames."""
        if "samplename" not in list(files_info_no_defaults):
            files_info_no_defaults["samplename"] = [
                f.split("_")[0] for f in files_info_no_defaults.index.tolist()
            ]
        if "derivatized" not in list(files_info_no_defaults):
            files_info_no_defaults["derivatized"] = False
        if "calibration_file" not in list(files_info_no_defaults):
            files_info_no_defaults["calibration_file"] = False

        for col in Project.files_info_defauls_columns:
            if col not in list(files_info_no_defaults):
                files_info_no_defaults[col] = 1
        return files_info_no_defaults

    def load_all_files(self):
        """Loads all files listed in 'files_info' into a dictionary, where keys are
        filenames. Each file is processed to clean and standardize data. It updates the
        'files' attribute with data frames of file contents and 'is_files_deriv' with
        derivative information. Marks 'files_loaded' as True after loading."""
        print("Info: load_all_files: loop started")
        if not self.files_info_created:
            self.load_files_info()
        for filename, is_deriv in zip(
            self.files_info.index, self.files_info["derivatized"]
        ):
            file = self.load_single_file(filename)
            self.files[filename] = file
            self.is_files_deriv[filename] = is_deriv
        self.files_loaded = True
        print("Info: load_all_files: files loaded")
        return self.files, self.is_files_deriv

    def load_single_file(self, filename):
        """Loads a single GCMS file by its name, cleans, and processes the data according
        to project settings (e.g., delimiter, columns to keep). It sums areas for duplicated
        compound names and handles dilution factors. Updates the file's data with iupac names
        and reorders columns. Logs the process and returns the cleaned DataFrame."""
        file = pd.read_csv(
            plib.Path(Project.in_path, filename + ".txt"),
            delimiter=Project.load_delimiter,
            index_col=0,
            skiprows=Project.load_skiprows,
        )
        columns_to_drop = [
            cl for cl in file.columns if cl not in Project.columns_to_keep_in_files
        ]
        file.drop(columns_to_drop, axis=1, inplace=True)
        file.rename(Project.columns_to_rename_in_files, inplace=True, axis="columns")

        file["comp_name"] = file["comp_name"].fillna("unidentified")
        sum_areas_in_file = file.groupby("comp_name")["area"].sum()
        # the first ret time is kept for each duplicated Name
        file.drop_duplicates(subset="comp_name", keep="first", inplace=True)
        file.set_index("comp_name", inplace=True)  # set the cas as the index
        file["area"] = sum_areas_in_file  # used summed areas as areas

        file["area_if_undiluted"] = (
            file["area"] * self.files_info.loc[filename, "dilution_factor"]
        )
        file["iupac_name"] = "n.a."
        new_cols_order = ["iupac_name"] + [
            col for col in file.columns if col != "iupac_name"
        ]
        file = file[new_cols_order]
        file.index.name = filename
        file.index = file.index.map(lambda x: x.lower())
        file.rename(Project.compounds_to_rename, inplace=True)
        print("\tInfo: load_single_file ", filename)
        return file

    def load_class_code_frac(self):
        """Loads the 'classifications_codes_fractions.xlsx' file containing information
        on SMARTS classifications. It first searches in the project's input path, then
        in the shared path. It logs the status and returns the DataFrame containing
        classification codes and fractions."""
        try:  # first try to find the file in the folder
            self.class_code_frac = pd.read_excel(
                plib.Path(Project.in_path, "classifications_codes_fractions.xlsx")
            )
            print("Info: load_class_code_frac: classifications_codes_fractions loaded")
        except FileNotFoundError:  # then try in the common input folder
            try:
                self.class_code_frac = pd.read_excel(
                    plib.Path(
                        Project.shared_path, "classifications_codes_fractions.xlsx"
                    )
                )
                print(
                    "Info: load_class_code_frac: classifications_codes_fractions loaded from"
                    + "shared folder (up one level)"
                )
            except FileNotFoundError:
                print(
                    'ERROR: the file "classifications_codes_fractions.xlsx" was not found ',
                    "look in example/data for a template",
                )
        all_classes = self.class_code_frac.classes.tolist()
        codes = self.class_code_frac.codes.tolist()  # list of code for each class
        mfs = self.class_code_frac.mfs.tolist()  # list of mass fraction of each class
        self.dict_classes_to_codes = dict(zip(all_classes, codes))  # dictionaries
        self.dict_classes_to_mass_fractions = dict(
            zip(all_classes, mfs)
        )  # dictionaries
        return self.class_code_frac

    def load_calibrations(self):
        """Loads calibration data from Excel files specified in the 'files_info' DataFrame,
        handles missing files, and coerces non-numeric values to NaN in calibration data
        columns. It ensures each calibration file is loaded once, updates the 'calibrations'
        attribute with calibration data, and sets 'calibrations_loaded' and
        'calibrations_not_present' flags based on the presence of calibration files."""
        if not self.files_info_created:
            self.load_files_info()
        if any(self.files_info["calibration_file"]):
            _files_info = self.files_info.drop_duplicates(subset="calibration_file")
            for cal_name, is_cal_deriv in zip(
                _files_info["calibration_file"], _files_info["derivatized"]
            ):
                try:
                    cal_file = pd.read_excel(
                        plib.Path(Project.in_path, cal_name + ".xlsx"), index_col=0
                    )
                except FileNotFoundError:
                    try:
                        cal_file = pd.read_excel(
                            plib.Path(Project.shared_path, cal_name + ".xlsx"),
                            index_col=0,
                        )
                    except FileNotFoundError:
                        print(
                            "ERROR: ",
                            cal_name,
                            ".xlsx not found in project nor shared path",
                        )
                cal_file.index.name = "comp_name"
                cols_cal_area = [c for c in list(cal_file) if "Area" in c]
                cols_cal_ppms = [c for c in list(cal_file) if "PPM" in c]
                cal_file[cols_cal_area + cols_cal_ppms] = cal_file[
                    cols_cal_area + cols_cal_ppms
                ].apply(pd.to_numeric, errors="coerce")
                if "iupac_name" not in list(cal_file):
                    for comp in cal_file.index.tolist():
                        cal_file.loc[comp, "iupac_name"] = get_iupac_from_pcp(comp)
                new_cols_order = ["iupac_name"] + [
                    col for col in cal_file.columns if col != "iupac_name"
                ]
                cal_file = cal_file[new_cols_order]
                cal_file.index = cal_file.index.map(lambda x: x.lower())
                self.calibrations[cal_name] = cal_file
                self.is_calibrations_deriv[cal_name] = is_cal_deriv
            self.calibrations_loaded = True
            self.iupac_to_calibrations_added = True
            self.calibrations_not_present = False
            print("Info: load_calibrations: calibarions loaded")
        else:
            self.calibrations_loaded = True
            self.calibrations_not_present = True
            print("Info: load_calibrations: no calibarions specified")

        return self.calibrations, self.is_calibrations_deriv

    def create_list_of_all_compounds(self):
        """Compiles a list of all unique compounds across all loaded files and calibrations,
        only for underivatized compounds. It ensures all files
        are loaded before compiling the list, excludes 'unidentified' compounds, and updates
        the 'list_of_all_compounds' attribute. Logs completion and returns the list."""
        if not self.files_loaded:
            self.load_all_files()
        if not self.calibrations_loaded:
            self.load_calibrations()
        _dfs: list[pd.DataFrame] = []
        for filename, file in self.files.items():
            if not self.is_files_deriv[filename]:
                _dfs.append(file)
        for filename, file in self.calibrations.items():
            if not self.is_calibrations_deriv[filename]:
                _dfs.append(file)
        # non-derivatized compounds
        all_compounds: pd.DataFrame = pd.concat(_dfs)

        set_of_all_compounds = pd.Index(all_compounds.index.unique())
        # Using set comprehension to remove unwanted elements
        filtered_compounds = {
            compound.strip()  # Remove leading/trailing spaces
            for compound in set_of_all_compounds
            if compound not in ["unidentified", None, False, "", " ", "''"]
        }
        # Converting the filtered set to a list
        self.list_of_all_compounds = list(filtered_compounds)
        self.list_of_all_compounds_created = True
        print(
            f"Info: create_list_of_all_compounds: list_of_all_compounds created {len(self.list_of_all_compounds) = }"
        )
        return self.list_of_all_compounds

    def create_list_of_all_deriv_compounds(self):
        """Compiles a list of all unique derivatized compounds across all loaded
        files and calibrations, adjusting compound names for derivatization indicators.
        Updates and returns the 'list_of_all_deriv_compounds' attribute."""
        if not self.files_loaded:
            self.load_all_files()
        if not self.calibrations_loaded:
            self.load_calibrations()
        _dfs_deriv: list[pd.DataFrame] = []
        for filename, file in self.files.items():
            if self.is_files_deriv[filename]:
                _dfs_deriv.append(file)
        add_to_idx = ", " + Project.string_in_deriv_names[0]
        for filename, file in self.calibrations.items():
            temporary = file.copy()
            if self.is_calibrations_deriv[filename]:
                # need to add to calib index to match file names
                temporary.index = temporary.index.map(lambda x: x + add_to_idx)
                _dfs_deriv.append(temporary)
        all_deriv_compounds: pd.DataFrame = pd.concat(_dfs_deriv)
        set_of_all_deriv_compounds = pd.Index(all_deriv_compounds.index.unique())
        filtered_deriv_compounds = {
            compound.strip()  # Remove leading/trailing spaces
            for compound in set_of_all_deriv_compounds
            if compound not in ["unidentified", None, False, "", " ", "''"]
        }
        # Converting the filtered set to a list
        self.list_of_all_deriv_compounds = list(filtered_deriv_compounds)
        self.list_of_all_deriv_compounds_created = True
        print(
            f"Info: create_list_of_all_deriv_compounds: list_of_all_deriv_compounds created {len(self.list_of_all_deriv_compounds) = }"
        )
        return self.list_of_all_deriv_compounds

    def load_compounds_properties(self):
        """Attempts to load the 'compounds_properties.xlsx' file containing physical
        and chemical properties of compounds. If not found, it creates a new properties
        DataFrame and updates the 'compounds_properties_created' attribute."""
        compounds_properties_path = plib.Path(
            Project.in_path, "compounds_properties.xlsx"
        )
        if compounds_properties_path.exists():
            cpdf = pd.read_excel(
                compounds_properties_path,
                index_col="comp_name",
            )
            # cpdf = _order_columns_in_compounds_properties(cpdf)
            # cpdf = cpdf.fillna(0)
            self.compounds_properties = cpdf
            self.compounds_properties_created = True
            print("Info: compounds_properties loaded")
        else:
            print("Warning: compounds_properties.xlsx not found, creating it")
            cpdf = self.create_compounds_properties()
        return self.compounds_properties

    def load_deriv_compounds_properties(self):
        """Attempts to load the 'deriv_compounds_properties.xlsx' file containing properties
        for derivatized compounds. If not found, it creates a new properties DataFrame
        for derivatized compounds and updates the 'deriv_compounds_properties_created' attribute.
        """
        compounds_deriv_properties_path = plib.Path(
            Project.in_path, "deriv_compounds_properties.xlsx"
        )
        if compounds_deriv_properties_path.exists():
            dcpdf = pd.read_excel(
                compounds_deriv_properties_path,
                index_col="comp_name",
            )
            # dcpdf = _order_columns_in_compounds_properties(dcpdf)
            # cpdf = dcpdf.fillna(0)
            self.deriv_compounds_properties = dcpdf
            self.deriv_compounds_properties_created = True
            print("Info: deriv_compounds_properties loaded")
        else:
            print("Warning: deriv_compounds_properties.xlsx not found, creating it")
            dcpdf = self.create_deriv_compounds_properties()
        return self.deriv_compounds_properties

    def create_compounds_properties(self):
        """Retrieves and organizes properties for underivatized compounds using pubchempy,
        updating the 'compounds_properties' attribute and saving the properties
        to 'compounds_properties.xlsx'."""
        print("Info: create_compounds_properties: started")

        if not self.class_code_frac_loaded:
            self.load_class_code_frac()
        if not self.list_of_all_compounds_created:
            self.create_list_of_all_compounds()
        # cpdf = pd.DataFrame(index=pd.Index(self.list_of_all_compounds))
        #
        cpdf = pd.DataFrame()
        print("Info: create_compounds_properties: looping over names")
        for name in self.list_of_all_compounds:
            cpdf = name_to_properties(
                comp_name=name,
                dict_classes_to_codes=self.dict_classes_to_codes,
                dict_classes_to_mass_fractions=self.dict_classes_to_mass_fractions,
                df=cpdf,
            )
        # cpdf = self._order_columns_in_compounds_properties(cpdf)
        # cpdf = cpdf.fillna(0)
        cpdf.index.name = "comp_name"
        self.compounds_properties = cpdf
        self.compounds_properties_created = True
        # save db in the project folder in the input
        cpdf.to_excel(plib.Path(Project.in_path, "compounds_properties.xlsx"))
        print(
            "Info: create_compounds_properties: compounds_properties created and saved"
        )
        return self.compounds_properties

    def create_deriv_compounds_properties(self):
        """Retrieves and organizes properties for derivatized compounds using pubchempy,
        linking them to their underivatized forms, updating the
        'deriv_compounds_properties' attribute, and saving the properties
        to 'deriv_compounds_properties.xlsx'."""
        if not self.class_code_frac_loaded:
            self.load_class_code_frac()
        if not self.list_of_all_deriv_compounds_created:
            self.create_list_of_all_deriv_compounds()
        deriv_to_underiv = {}
        for derivname in self.list_of_all_deriv_compounds:
            parts = derivname.split(",")
            is_der_str_in_part2: bool = any(
                [
                    der_str in parts[-1].strip()
                    for der_str in Project.string_in_deriv_names
                ]
            )
            if len(parts) > 1 and is_der_str_in_part2:
                # If the suffix is a known derivatization, use the part before the comma
                deriv_to_underiv[derivname] = ",".join(parts[:-1])
            else:
                # In all other cases, mark as "unidentified"
                deriv_to_underiv[derivname] = "unidentified"
        print("Info: create_deriv_compounds_properties: looping over names")
        underiv_comps_to_search_for = [
            c for c in deriv_to_underiv.values() if c != "unidentified"
        ]
        dcpdf = pd.DataFrame()
        for name in underiv_comps_to_search_for:
            dcpdf = name_to_properties(
                comp_name=name,
                dict_classes_to_codes=self.dict_classes_to_codes,
                dict_classes_to_mass_fractions=self.dict_classes_to_mass_fractions,
                df=dcpdf,
            )
        dcpdf.index.name = "underiv_comp_name"
        dcpdf.reset_index(inplace=True)
        underiv_to_deriv = {
            v: k for k, v in deriv_to_underiv.items() if v != "unidentified"
        }
        # Add a new column for the derivatized compound names
        # If a name is not in the underiv_to_deriv (thus 'unidentified'), it will get a value of NaN

        dcpdf["comp_name"] = dcpdf["underiv_comp_name"].apply(
            lambda x: underiv_to_deriv.get(x, "unidentified")
        )
        dcpdf.set_index("comp_name", inplace=True)
        # save db in the project folder in the input
        self.deriv_compounds_properties = dcpdf
        dcpdf.to_excel(plib.Path(Project.in_path, "deriv_compounds_properties.xlsx"))
        self.compounds_properties_created = True
        print(
            "Info: create_deriv_compounds_properties: deriv_compounds_properties created and saved"
        )
        return self.deriv_compounds_properties

    # def add_iupac_to_calibrations(self):
    #     """Adds the IUPAC name to each compound in the calibration data,
    #     istinguishing between underivatized and derivatized calibrations,
    #     and updates the corresponding calibration dataframes."""
    #     if not self.calibrations_loaded:
    #         self.load_calibrations()
    #     if not self.compounds_properties_created:
    #         self.load_compounds_properties()
    #     if self.deriv_files_present:
    #         if not self.deriv_compounds_properties_created:
    #             self.load_deriv_compounds_properties()
    #     for calibname, calib in self.calibrations.items():
    #         if not self.is_calibrations_deriv[calibname]:
    #             df_comps = self.compounds_properties
    #             for c in calib.index.tolist():
    #                 iup = df_comps.loc[c, "iupac_name"]
    #                 calib.loc[c, "iupac_name"] = iup
    #         else:
    #             df_comps = self.deriv_compounds_properties
    #             df_comps.set_index("underiv_comp_name", inplace=True)
    #             for c in calib.index.tolist():
    #                 iup = df_comps.loc[c, "iupac_name"]
    #                 try:

    #                     calib.loc[c, "iupac_name"] = iup
    #                 except ValueError:
    #                     print(f"Calibration could not set iupac value for {c}")

    #     self.iupac_to_calibrations_added = True
    #     return self.calibrations, self.is_calibrations_deriv

    def add_iupac_to_files(self):
        """Adds the IUPAC name to each compound in the loaded files,
        distinguishing between underivatized and derivatized compounds,
        and updates the corresponding file dataframes."""
        if not self.files_loaded:
            self.load_all_files()
        if not self.compounds_properties_created:
            self.load_compounds_properties()
        if self.deriv_files_present:
            if not self.deriv_compounds_properties_created:
                self.load_deriv_compounds_properties()
        for filename, file in self.files.items():
            if not self.is_files_deriv[filename]:
                df_comps = self.compounds_properties
            else:
                df_comps = self.deriv_compounds_properties
            for c in file.index.tolist():
                if c == "unidentified":
                    file.loc[c, "iupac_name"] = "unidentified"
                else:
                    try:
                        iup = df_comps.loc[c, "iupac_name"]
                    except KeyError:
                        iup = "unidentified"
                    file.loc[c, "iupac_name"] = iup
        self.iupac_to_files_added = True
        return self.files, self.is_files_deriv

    def apply_calibration_to_files(self):
        """Applies the appropriate calibration curve to each compound
        in the loaded files, adjusting concentrations based on calibration
        data, and updates the 'files' attribute with calibrated data."""
        print("Info: apply_calibration_to_files: loop started")
        if not self.files_loaded:
            self.load_all_files()
        if not self.calibrations_loaded:
            self.load_calibrations()
        if self.calibrations_not_present:
            print(
                "WARNING: apply_calibration_to_files, no calibration is available",
                "files are unchanged",
            )
            return self.files, self.is_files_deriv
        if not self.iupac_to_files_added:
            _, _ = self.add_iupac_to_files()

        for filename, _ in self.files.items():
            calibration_name = self.files_info.loc[filename, "calibration_file"]
            calibration = self.calibrations[calibration_name]
            if not self.is_files_deriv[filename]:
                df_comps = self.compounds_properties
            else:
                df_comps = self.deriv_compounds_properties
            file = self._apply_calib_to_file(filename, calibration, df_comps)
            if Project.auto_save_to_excel:
                self.save_file(file, filename)
        self.calibration_to_files_applied = True
        return self.files, self.is_files_deriv

    def _apply_calib_to_file(self, filename, calibration, df_comps):
        """computes conc data based on the calibration provided.
        If semi_calibration is specified, the closest compound in terms of
        Tanimoto similarity and molecular weight similarity is used for
        compounds where a calibration entry is not available"""
        # """calibration.rename(Project.compounds_to_rename, inplace=True)"""
        # print(file)
        print("\tInfo: _apply_calib_to_file ", filename)
        clbrtn = calibration.set_index("iupac_name")
        cpmnds = df_comps.set_index("iupac_name")
        cpmnds = cpmnds[~cpmnds.index.duplicated(keep="first")].copy()
        cols_cal_area = [c for c in list(calibration) if "Area" in c]
        cols_cal_ppms = [c for c in list(calibration) if "PPM" in c]
        tot_sample_conc = self.files_info.loc[
            filename, "total_sample_conc_in_vial_mg_L"
        ]
        sample_yield_feed_basis = self.files_info.loc[
            filename, "sample_yield_on_feedstock_basis_fr"
        ]
        for comp, iupac in zip(
            self.files[filename].index.tolist(),
            self.files[filename]["iupac_name"].tolist(),
        ):
            if (
                comp == "unidentified"
                or iupac == "unidentified"
                or comp == "n.a."
                or iupac == "n.a."
            ):
                conc_mg_l = np.nan
                comps_for_calib = "n.a."
            else:
                if iupac in clbrtn.index.tolist():
                    # areas and ppms for the calibration are taken from df_clbr
                    cal_areas = clbrtn.loc[iupac, cols_cal_area].to_numpy(dtype=float)
                    cal_ppms = clbrtn.loc[iupac, cols_cal_ppms].to_numpy(dtype=float)
                    # linear fit of calibration curve (exclude nan),
                    # get ppm from area
                    fit = np.polyfit(
                        cal_areas[~np.isnan(cal_areas)],
                        cal_ppms[~np.isnan(cal_ppms)],
                        1,
                    )
                    # concentration at the injection solution (GC vial)
                    # ppp = mg/L
                    conc_mg_l = np.poly1d(fit)(self.files[filename].loc[comp, "area"])
                    if conc_mg_l < 0:
                        conc_mg_l = 0
                    comps_for_calib = "self"
                else:
                    if not Project.semi_calibration:
                        conc_mg_l = np.nan
                        comps_for_calib = "n.a."
                        continue
                    # get property of the compound as first elements
                    mws = [cpmnds.loc[iupac, "molecular_weight"]]
                    smis = [cpmnds.loc[iupac, "canonical_smiles"]]
                    names_cal = [iupac]
                    # then add all properties for all calibrated compounds
                    # if the sample was not derivatized (default)
                    # if not self.is_files_deriv[filename]:
                    for c in clbrtn.index.tolist():
                        names_cal.append(c)
                        # print(df_comps.index)
                        try:
                            smis.append(cpmnds.loc[c, "canonical_smiles"])
                            mws.append(cpmnds.loc[c, "molecular_weight"])
                        except KeyError:
                            print(f"inisde calib {c = }")
                    # calculate the delta mw with all calib compounds
                    delta_mw = np.abs(np.asarray(mws)[0] - np.asarray(mws)[1:])
                    # get mols and fingerprints from rdkit for each comp
                    mols = [Chem.MolFromSmiles(smi) for smi in smis]
                    fps = [
                        GetMorganFingerprintAsBitVect(ml, 2, nBits=1024) for ml in mols
                    ]
                    # perform Tanimoto similarity betwenn the first and all
                    # other compounds
                    s = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
                    # create a df with results
                    df_sim = pd.DataFrame(
                        data={
                            "name": names_cal[1:],
                            "smiles": smis[1:],
                            "Similarity": s,
                            "delta_mw": delta_mw,
                        }
                    )
                    # put the index title as the comp
                    df_sim.index.name = iupac
                    # sort values based on similarity and delta mw
                    df_sim = df_sim.sort_values(
                        ["Similarity", "delta_mw"], ascending=[False, True]
                    )
                    # remove values below thresholds
                    df_sim = df_sim[
                        df_sim.Similarity >= Project.tanimoto_similarity_threshold
                    ]
                    df_sim = df_sim[
                        df_sim.delta_mw < Project.delta_mol_weight_threshold
                    ]
                    # if a compound matches the requirements
                    if not df_sim.empty:  # assign the calibration
                        name_clbr = df_sim.name.tolist()[0]

                        # areas and ppms are taken from df_clbr
                        cal_areas = clbrtn.loc[name_clbr, cols_cal_area].to_numpy(
                            dtype=float
                        )
                        cal_ppms = clbrtn.loc[name_clbr, cols_cal_ppms].to_numpy(
                            dtype=float
                        )
                        # linear fit of calibration curve (exclude nan),
                        # get ppm from area
                        fit = np.polyfit(
                            cal_areas[~np.isnan(cal_areas)],
                            cal_ppms[~np.isnan(cal_ppms)],
                            1,
                        )
                        # concentration at the injection solution (GC vial)
                        # ppm = mg/L
                        conc_mg_l = np.poly1d(fit)(
                            self.files[filename].loc[comp, "area"]
                        )
                        if conc_mg_l < 0:
                            conc_mg_l = 0
                        # note type of calibration and compound used
                        comps_for_calib = (
                            name_clbr
                            + " (sim="
                            + str(round(df_sim.Similarity.values[0], 2))
                            + "; dwt="
                            + str(int(df_sim.delta_mw.values[0]))
                            + ")"
                        )
                    else:  # put concentrations to nan
                        conc_mg_l = np.nan
                        comps_for_calib = "n.a."
            self.files[filename].loc[comp, "conc_vial_mg_L"] = conc_mg_l
            self.files[filename].loc[comp, "conc_vial_if_undiluted_mg_L"] = (
                conc_mg_l * self.files_info.loc[filename, "dilution_factor"]
            )
            self.files[filename].loc[comp, "fraction_of_sample_fr"] = (
                conc_mg_l / tot_sample_conc
            )
            self.files[filename].loc[comp, "fraction_of_feedstock_fr"] = (
                conc_mg_l / tot_sample_conc * sample_yield_feed_basis
            )
            self.files[filename].loc[
                comp, "compound_used_for_calibration"
            ] = comps_for_calib
        if np.isnan(self.files[filename]["conc_vial_mg_L"]).all():
            print(
                f"WARNING: the file {filename} does not contain any ",
                "compound for which a calibration nor a semicalibration is available.",
                "\n either lower similarity thresholds, add calibration compounds, or",
                "calibration_file=False in files_info.xlsx",
            )
        return self.files[filename]

    def add_stats_to_files_info(self):
        """Computes and adds statistical data for each file to the 'files_info'
        DataFrame, such as maximum height, area, and concentrations,
        updating the 'files_info' with these statistics."""
        print("Info: add_stats_to_files_info: started")

        if not self.calibration_to_files_applied:
            self.apply_calibration_to_files()
        if not self.calibrations_not_present:  # calinrations available
            numeric_columns = [
                "height",
                "area",
                "area_if_undiluted",
                "conc_vial_mg_L",
                "conc_vial_if_undiluted_mg_L",
                "fraction_of_sample_fr",
                "fraction_of_feedstock_fr",
            ]
        else:
            numeric_columns = ["height", "area", "area_if_undiluted"]
        max_columns = [f"max_{nc}" for nc in numeric_columns]
        total_columns = [f"total_{nc}" for nc in numeric_columns]
        for name, df in self.files.items():
            for ncol, mcol, tcol in zip(numeric_columns, max_columns, total_columns):
                self.files_info.loc[name, mcol] = df[ncol].max()
                self.files_info.loc[name, tcol] = df[ncol].sum()
        for name, df in self.files.items():
            self.files_info.loc[name, "compound_with_max_area"] = df[
                df["area"] == df["area"].max()
            ].index[0]
            if not self.calibrations_not_present:
                self.files_info.loc[name, "compound_with_max_conc"] = df[
                    df["conc_vial_mg_L"]
                    == self.files_info.loc[name, "max_conc_vial_mg_L"]
                ].index[0]
        # convert max and total columns to float
        for col in max_columns + total_columns:
            if col in self.files_info.columns:
                self.files_info[col] = self.files_info[col].astype(float)

        if Project.auto_save_to_excel:
            self.save_files_info()
        self.stats_to_files_info_added = True
        return self.files_info

    def create_samples_info(self):
        """Creates a summary 'samples_info' DataFrame from 'files_info',
        aggregating data for each sample, and updates the 'samples_info'
        attribute with this summarized data."""
        if not self.files_info_created:
            self.load_files_info()
        if not self.stats_to_files_info_added:
            self.add_stats_to_files_info()

        # Define numeric columns based on calibration presence
        if not self.calibrations_not_present:  # calibrations available
            numeric_columns = [
                "height",
                "area",
                "area_if_undiluted",
                "conc_vial_mg_L",
                "conc_vial_if_undiluted_mg_L",
                "fraction_of_sample_fr",
                "fraction_of_feedstock_fr",
            ]
        else:
            numeric_columns = ["height", "area", "area_if_undiluted"]
        files_info = self.files_info.reset_index()
        max_columns = [f"max_{nc}" for nc in numeric_columns]
        total_columns = [f"total_{nc}" for nc in numeric_columns]
        all_numeric_columns = numeric_columns + max_columns + total_columns
        # Ensure these columns are in files_info before proceeding
        numcol = [col for col in all_numeric_columns if col in files_info.columns]

        # Identify non-numeric columns
        non_numcol = [
            col
            for col in files_info.columns
            if col not in numcol and col != "samplename"
        ]

        # Initialize samples_info DataFrame
        # self.samples_info = pd.DataFrame(columns=self.files_info.columns)

        # Create an aggregation dictionary

        agg_dict = {
            **{nc: "mean" for nc in numcol},
            **{nnc: lambda x: list(x) for nnc in non_numcol},
        }
        agg_dict_std = {
            **{nc: "std" for nc in numcol},
            **{nnc: lambda x: list(x) for nnc in non_numcol},
        }

        # Group by 'samplename' and apply aggregation, make sure 'samplename' is not part of the aggregation
        _samples_info_std = files_info.groupby("samplename").agg(agg_dict_std)
        _samples_info = files_info.groupby("samplename").agg(agg_dict)

        self.samples_info = _samples_info[non_numcol + numcol]
        self.samples_info_std = _samples_info_std[non_numcol + numcol]
        self.samples_info_created = True
        if Project.auto_save_to_excel:
            self.save_samples_info()
        print("Info: create_samples_info: samples_info created")
        return self.samples_info, self.samples_info_std

    def create_samples_from_files(self):
        """Generates a DataFrame for each sample by averaging and calculating
        the standard deviation of replicates, creating a comprehensive
        dataset for each sample in the project."""
        if not self.samples_info_created:
            _, _ = self.create_samples_info()
        if not self.calibration_to_files_applied:
            self.apply_calibration_to_files()
        for samplename in self.samples_info.index:
            print("Sample: ", samplename)
            _files = []
            for filename in self.files_info.index[
                self.files_info["samplename"] == samplename
            ]:
                print("\tFile: ", filename)
                _files.append(self.files[filename])
            sample, sample_std = self._create_sample_from_files(_files, samplename)
            self.samples[samplename] = sample
            self.samples_std[samplename] = sample_std
            if Project.auto_save_to_excel:
                self.save_sample(sample, sample_std, samplename)
        self.samples_created = True
        return self.samples, self.samples_std

    def _create_sample_from_files(
        self, files_in_sample: list[pd.DataFrame], samplename: str
    ):
        """Creates a sample dataframe and a standard deviation dataframe from files
        that are replicates of the same sample. This process includes aligning dataframes,
        filling missing values, calculating averages and standard deviations,
        and merging non-numerical data."""
        all_ordered_columns = files_in_sample[0].columns.tolist()
        if not self.calibrations_not_present:
            non_num_columns = ["iupac_name", "compound_used_for_calibration"]
        else:
            non_num_columns = ["iupac_name"]
        # Step 1: Create a comprehensive index of all unique compounds
        all_compounds = pd.Index([])
        for df in files_in_sample:
            all_compounds = all_compounds.union(df.index)

        # Step 2: Align all DataFrames to the comprehensive index
        aligned_dfs: list[pd.DataFrame] = [
            df.reindex(all_compounds) for df in files_in_sample
        ]
        # aligned_dfs = [
        #     df.align(files_in_sample[0], join="outer", axis=0)[0]
        #     for df in files_in_sample
        # ]  # Align indices
        # Fill NaN values for numerical columns after alignment and before concatenation
        filled_dfs = [df.fillna(0.0) for df in aligned_dfs]
        # Keep non-numerical data separately and ensure no duplicates
        non_num_data: pd.DataFrame = pd.concat(
            [df[non_num_columns].drop_duplicates() for df in files_in_sample]
        ).drop_duplicates()
        # Separating numerical data to fill NaNs with zeros
        num_data_filled = [df.drop(columns=non_num_columns) for df in filled_dfs]
        # Calculating the average and std for numerical data
        sample = pd.concat(num_data_filled).groupby(level=0).mean().astype(float)
        sample_std = pd.concat(num_data_filled).groupby(level=0).std().astype(float)
        # Merging non-numerical data with the numerical results
        sample = sample.merge(
            non_num_data, left_index=True, right_index=True, how="left"
        )
        sample_std = sample_std.merge(
            non_num_data, left_index=True, right_index=True, how="left"
        )
        sample = sample.sort_values(by=Project.column_to_sort_values_in_samples)
        # Apply the same order to 'sample_std' using reindex
        sample_std = sample_std.reindex(sample.index)
        sample = sample[all_ordered_columns]
        sample_std = sample_std[all_ordered_columns]
        sample.index.name = samplename
        sample_std.index.name = samplename

        return sample, sample_std

    # def add_stats_to_samples_info(self):
    #     """Generates summary statistics for each sample based on the processed files,
    #     adding these statistics to the 'samples_info' DataFrame.
    #     Updates the 'samples_info' with sample-specific maximum,
    #     total values, and compound with maximum concentration."""
    #     print("Info: add_stats_to_samples_info: started")
    #     if not self.samples_created:
    #         self.create_samples_from_files()
    #     if not self.samples_info_created:
    #         self.create_samples_info()
    #     if not self.calibrations_not_present:  # calibrations available
    #         numeric_columns = [
    #             "height",
    #             "area",
    #             "area_if_undiluted",
    #             "conc_vial_mg_L",
    #             "conc_vial_if_undiluted_mg_L",
    #             "fraction_of_sample_fr",
    #             "fraction_of_feedstock_fr",
    #         ]
    #     else:
    #         numeric_columns = ["height", "area", "area_if_undiluted"]
    #     max_columns = [f"max_{nc}" for nc in numeric_columns]
    #     total_columns = [f"total_{nc}" for nc in numeric_columns]
    #     for name, df in self.samples.items():
    #         for ncol, mcol, tcol in zip(numeric_columns, max_columns, total_columns):
    #             self.samples_info.loc[name, mcol] = df[ncol].max()
    #             self.samples_info.loc[name, tcol] = df[ncol].sum()
    #     for name, df in self.samples.items():
    #         self.samples_info.loc[name, "compound_with_max_area"] = df[
    #             df["area"] == df["area"].max()
    #         ].index[0]
    #         if not self.calibrations_not_present:
    #             self.samples_info.loc[name, "compound_with_max_conc"] = df[
    #                 df["conc_vial_mg_L"]
    #                 == self.samples_info.loc[name, "max_conc_vial_mg_L"]
    #             ].index[0]
    #     # convert max and total columns to float
    #     for col in max_columns + total_columns:
    #         if col in self.samples_info.columns:
    #             try:
    #                 self.samples_info[col] = self.samples_info[col].astype(float)
    #             except ValueError:
    #                 print(self.samples_info[col])
    #     self.stats_to_samples_info_added = True
    #     if Project.auto_save_to_excel:
    #         self.save_samples_info()
    #     self.stats_to_samples_info_added = True
    #     return self.samples_info

    def create_files_param_report(self, param="conc_vial_mg_L"):
        """Creates a detailed report for each parameter across all FILES,
        displaying the concentration of each compound in each sample.
        This report aids in the analysis and comparison of compound
        concentrations across FILES."""
        print("Info: create_files_param_report: ", param)
        if param not in Project.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if not self.calibration_to_files_applied:
            self.apply_calibration_to_files()
        rep_columns = self.files_info.index.tolist()
        _all_comps = self.compounds_properties["iupac_name"].tolist()
        if self.deriv_files_present:
            _all_comps += self.deriv_compounds_properties["iupac_name"].tolist()
        rep_index = list(set(_all_comps))
        rep = pd.DataFrame(index=rep_index, columns=rep_columns, dtype="float")
        rep.index.name = param

        for comp in rep.index.tolist():  # add conc values
            for name in rep.columns.tolist():
                smp = self.files[name].set_index("iupac_name")
                smp = smp[~smp.index.duplicated(keep="first")]
                try:
                    rep.loc[comp, name] = smp.loc[comp, param]
                except KeyError:
                    rep.loc[comp, name] = 0

        rep = rep.sort_index(key=rep.max(1).get, ascending=False)
        rep = rep.loc[:, rep.any(axis=0)]  # drop columns with only 0s
        rep = rep.loc[rep.any(axis=1), :]  # drop rows with only 0s
        self.files_reports[param] = rep
        self.list_of_files_param_reports.append(param)
        if Project.auto_save_to_excel:
            self.save_files_param_report(param=param)
        return rep

    def create_files_param_aggrrep(self, param="conc_vial_mg_L"):
        """Aggregates compound concentration data by functional group for each
        parameter across all FILES, providing a summarized view of functional
        group concentrations. This aggregation facilitates the understanding
        of functional group distribution across FILES."""
        print("Info: create_param_aggrrep: ", param)
        if param not in Project.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.list_of_files_param_reports:
            self.create_files_param_report(param)

        # fg = functional groups, mf = mass fraction
        filenames = self.files_info.index.tolist()
        _all_comps = self.files_reports[param].index.tolist()
        cols_with_fg_mf_labs = list(self.compounds_properties)
        if self.deriv_files_present:
            for c in list(self.deriv_compounds_properties):
                if c not in cols_with_fg_mf_labs:
                    cols_with_fg_mf_labs.append(c)
        fg_mf_labs = [
            c
            for c in cols_with_fg_mf_labs
            if c.startswith("fg_mf_")
            if c != "fg_mf_total"
        ]
        fg_labs = [c[6:] for c in fg_mf_labs]
        # create a df with iupac name index and fg_mf columns (underiv and deriv)
        comps_df = self.compounds_properties.set_index("iupac_name")
        if self.deriv_files_present:
            deriv_comps_df = self.deriv_compounds_properties.set_index("iupac_name")
            all_comps_df = pd.concat([comps_df, deriv_comps_df])
        else:
            all_comps_df = comps_df
        all_comps_df = all_comps_df[~all_comps_df.index.duplicated(keep="first")]
        fg_mf_all = pd.DataFrame(index=_all_comps, columns=fg_mf_labs)
        for idx in fg_mf_all.index.tolist():
            fg_mf_all.loc[idx, fg_mf_labs] = all_comps_df.loc[idx, fg_mf_labs]
        # create the aggregated dataframes and compute aggregated results
        aggrrep = pd.DataFrame(columns=filenames, index=fg_labs, dtype="float")
        aggrrep.index.name = param  # is the parameter
        aggrrep.fillna(0, inplace=True)
        for col in filenames:
            list_iupac = self.files_reports[param].index
            signal = self.files_reports[param].loc[:, col].values
            for fg, fg_mf in zip(fg_labs, fg_mf_labs):
                # each compound contributes to the cumulative sum of each
                # functional group for the based on the mass fraction it has
                # of that functional group (fg_mf act as weights)
                # if fg_mf in subrep: multiply signal for weight and sum
                # to get aggregated
                weights = fg_mf_all.loc[list_iupac, fg_mf].astype(signal.dtype)

                aggrrep.loc[fg, col] = (signal * weights).sum()
        aggrrep = aggrrep.loc[(aggrrep != 0).any(axis=1), :]  # drop rows with only 0
        aggrrep = aggrrep.sort_index(key=aggrrep[filenames].max(1).get, ascending=False)
        self.files_aggrreps[param] = aggrrep
        self.list_of_files_param_aggrreps.append(param)
        if Project.auto_save_to_excel:
            self.save_files_param_aggrrep(param=param)
        return aggrrep

    def create_samples_param_report(self, param: str = "conc_vial_mg_L"):
        print(f"Info: create_samples_param_report: {param = }")
        if param not in Project.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.list_of_files_param_reports:
            self.create_files_param_report(param)
        file_to_sample_rename = dict(
            zip(self.files_info.index.tolist(), self.files_info["samplename"])
        )
        filerep = self.files_reports[param].copy()
        filerep.rename(columns=file_to_sample_rename, inplace=True)
        self.samples_reports[param] = filerep.T.groupby(by=filerep.columns).mean().T
        self.samples_reports_std[param] = filerep.T.groupby(by=filerep.columns).std().T
        self.list_of_samples_param_reports.append(param)
        if Project.auto_save_to_excel:
            self.save_samples_param_report(param=param)
        return self.samples_reports[param], self.samples_reports_std[param]

    def create_samples_param_aggrrep(self, param: str = "conc_vial_mg_L"):
        print(f"Info: create_samples_param_aggrrep: {param = }")
        if param not in Project.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.list_of_files_param_aggrreps:
            self.create_files_param_aggrrep(param)
        file_to_sample_rename = dict(
            zip(self.files_info.index.tolist(), self.files_info["samplename"])
        )
        fileagg = self.files_aggrreps[param].copy()
        fileagg.rename(columns=file_to_sample_rename, inplace=True)
        self.samples_aggrreps[param] = fileagg.T.groupby(by=fileagg.columns).mean().T
        self.samples_aggrreps_std[param] = fileagg.T.groupby(by=fileagg.columns).std().T
        self.list_of_samples_param_aggrreps.append(param)
        if Project.auto_save_to_excel:
            self.save_samples_param_aggrrep(param=param)
        return self.samples_aggrreps[param], self.samples_aggrreps_std[param]

    def save_files_info(self):
        """Saves the 'files_info' DataFrame as an Excel file in a 'files'
        subfolder within the project's output path,
        facilitating easy access to and sharing of file metadata."""
        out_path = plib.Path(Project.out_path, "files")
        out_path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_excel(plib.Path(out_path, "files_infos.xlsx"))
        print("Info: save_files_info: files_info saved")

    def save_file(self, file, filename):
        """Saves an individual file's DataFrame as an Excel file in a 'files'
        subfolder, using the filename as the Excel file's name,
        allowing for detailed inspection of specific file data."""
        out_path = plib.Path(Project.out_path, "files")
        out_path.mkdir(parents=True, exist_ok=True)
        file.to_excel(plib.Path(out_path, filename + ".xlsx"))
        print("Info: save_files: ", filename, " saved")

    def save_samples_info(self):
        """Saves the 'samples_info' DataFrame as an Excel file in a 'samples'
        subfolder within the project's output path, after ensuring that sample
        statistics have been added, providing a summarized view of sample data."""
        if not self.samples_info_created:
            self.create_samples_info()
        out_path = plib.Path(Project.out_path, "samples")
        out_path.mkdir(parents=True, exist_ok=True)
        self.samples_info.to_excel(plib.Path(out_path, "samples_info.xlsx"))
        self.samples_info_std.to_excel(plib.Path(out_path, "samples_info_std.xlsx"))
        print("Info: save_samples_info: samples_info saved")

    def save_sample(self, sample, sample_std, samplename):
        """Saves both the sample and its standard deviation DataFrames
        as Excel files in a 'samples' subfolder, using the sample name and
        appending '_std' for the standard deviation file,
        offering a detailed and standardized view of sample data and variability."""
        out_path = plib.Path(Project.out_path, "samples")
        out_path.mkdir(parents=True, exist_ok=True)
        sample.to_excel(plib.Path(out_path, samplename + ".xlsx"))
        sample_std.to_excel(plib.Path(out_path, samplename + "_std.xlsx"))
        print("Info: save_sample: ", samplename, "saved")

    def save_files_param_report(self, param="conc_inj_mg_L"):
        """Saves a parameter-specific report for all files as an Excel
        file in a 'files_reports' subfolder, organizing data by
        the specified parameter to facilitate comprehensive analysis across files."""
        if param not in self.list_of_files_param_reports:
            self.create_files_param_report(param)
        name = "rep_files_" + param
        out_path = plib.Path(Project.out_path, "files_reports")
        out_path.mkdir(parents=True, exist_ok=True)
        self.files_reports[param].to_excel(plib.Path(out_path, name + ".xlsx"))
        print("Info: save_files_param_report: ", name, " saved")

    def save_files_param_aggrrep(self, param="conc_inj_mg_L"):
        """Saves a parameter-specific aggregated report for all files as an
        Excel file in an 'aggr_files_reports' subfolder, summarizing data by
        functional groups for the specified parameter, providing insights into
        the composition of samples at a higher level of abstraction."""
        if param not in self.list_of_files_param_aggrreps:
            self.create_files_param_aggrrep(param)
        name = "aggreg_files_rep_" + param
        out_path = plib.Path(Project.out_path, "aggr_files_reports")
        out_path.mkdir(parents=True, exist_ok=True)
        self.files_aggrreps[param].to_excel(plib.Path(out_path, name + ".xlsx"))
        print("Info: save_files_param_aggrrep: ", name, " saved")

    def save_samples_param_report(self, param="conc_inj_mg_L"):
        """Saves a parameter-specific report for all samples as an Excel
        file in a 'samples_reports' subfolder, along with a corresponding
        standard deviation report, enabling detailed analysis of parameter
        distribution across samples."""
        if param not in self.list_of_samples_param_reports:
            self.create_samples_param_report(param)
        name = "rep_samples_" + param
        out_path = plib.Path(Project.out_path, "samples_reports")
        out_path.mkdir(parents=True, exist_ok=True)
        self.samples_reports[param].to_excel(plib.Path(out_path, name + ".xlsx"))
        self.samples_reports_std[param].to_excel(
            plib.Path(out_path, name + "_std.xlsx")
        )
        print("Info: save_samples_param_report: ", name, " saved")

    def save_samples_param_aggrrep(self, param="conc_inj_mg_L"):
        """Saves a parameter-specific aggregated report for all samples as an Excel file
        in an 'aggr_samples_reports' subfolder, along with a standard deviation report,
        highlighting the functional group contributions to samples'
        composition for the specified parameter."""
        if param not in self.list_of_samples_param_aggrreps:
            self.create_samples_param_aggrrep(param)
        name = "aggreg_samples_rep_" + param
        out_path = plib.Path(Project.out_path, "aggr_samples_reports")
        out_path.mkdir(parents=True, exist_ok=True)
        self.samples_aggrreps[param].to_excel(plib.Path(out_path, name + ".xlsx"))
        self.samples_aggrreps_std[param].to_excel(
            plib.Path(out_path, name + "_std.xlsx")
        )
        print("Info: save_samples_param_aggrrep: ", name, " saved")
