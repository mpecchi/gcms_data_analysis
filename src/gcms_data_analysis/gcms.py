from __future__ import annotations
from typing import Literal
import pathlib as plib
import numpy as np
import pandas as pd
import pubchempy as pcp
import ele
from gcms_data_analysis.fragmenter import Fragmenter


class Project:
    """
    Represents a project (identified by the folder where the data is stored)
    for TGA data analysis.

    """

    def __init__(
        self,
        folder_path: plib.Path | str,
        name: str | None = None,
        apply_semi_calibration: bool = True,
        tanimoto_similarity_threshold: float = 0.4,
        delta_mol_weight_threshold: int = 100,
        file_load_skiprows: int = 8,
        file_load_delimiter: Literal["\t", ",", ";"] = "\t",
        file_load_format: Literal[".txt", ".csv"] = ".txt",
        column_to_sort_values_in_samples: Literal[
            "retention_time", "area", "height"
        ] = "retention_time",
        plot_font: Literal["Dejavu Sans", "Times New Roman"] = "Dejavu Sans",
        plot_grid: bool = False,
        auto_save_to_excel: bool = True,
        columns_to_rename_and_keep_in_files: dict[str, str] | None = None,
        compounds_to_rename_in_files: dict[str, str] | None = None,
        param_to_axis_label: dict[str, str] | None = None,
        string_in_deriv_names: list[str] | None = None,
    ):
        self.folder_path = plib.Path(folder_path)
        self.out_path = plib.Path(self.folder_path, "output")
        if name is None:
            self.name = self.folder_path.parts[-1]
        else:
            self.name = name
        self.apply_semi_calibration = apply_semi_calibration
        self.tanimoto_similarity_threshold = tanimoto_similarity_threshold
        self.delta_mol_weight_threshold = delta_mol_weight_threshold
        self.file_load_skiprows = file_load_skiprows
        self.file_load_delimiter = file_load_delimiter
        self.file_load_format = file_load_format
        self.column_to_sort_values_in_samples = column_to_sort_values_in_samples
        self.plot_font = plot_font
        self.plot_grid = plot_grid
        self.auto_save_to_excel = auto_save_to_excel

        if columns_to_rename_and_keep_in_files is None:
            self.columns_to_rename_and_keep_in_files = {
                "Ret.Time": "retention_time",
                "Height": "height",
                "Area": "area",
                "Name": "comp_name",
            }
        else:
            self.columns_to_rename_and_keep_in_files = (
                columns_to_rename_and_keep_in_files
            )
        if compounds_to_rename_in_files is None:
            self.compounds_to_rename_in_files = {}
        else:
            self.compounds_to_rename_in_files = compounds_to_rename_in_files
        if param_to_axis_label is None:
            self.param_to_axis_label = {
                "height": "Peak Height [-]",
                "area": "Peak Area [-]",
                "area_if_undiluted": "Peak Area [-]",
                "conc_vial_mg_L": "conc. [mg/L] (ppm)",
                "conc_vial_if_undiluted_mg_L": "conc. [mg/L] (ppm)",
                "fraction_of_sample_fr": "mass fraction [g/g$_{sample}$]",
                "fraction_of_feedstock_fr": "mass fraction [g/g$_{feedstock}$]",
            }
        else:
            self.param_to_axis_label = param_to_axis_label
        if string_in_deriv_names is None:
            self.string_in_deriv_names = [
                "deriv",
                "tms",
                "tbms",
                "trimethylsilyl",
            ]
        else:
            self.string_in_deriv_names = string_in_deriv_names
        # this does not depend on initialization, static default
        self.files_info_defauls_columns = [
            "dilution_factor",
            "total_sample_conc_in_vial_mg_L",
            "sample_yield_on_feedstock_basis_fr",
        ]

        self.files_info: pd.DataFrame | None = None
        self.class_code_frac: pd.DataFrame | None = None
        self.dict_classes_to_codes: dict[str, str] | None = None
        self.dict_classes_to_mass_fractions: dict[str, float] | None = None

        self.list_of_all_compounds: list[str] | None = None
        self.compounds_properties: pd.DataFrame | None = None
        self.dict_names_to_iupacs: dict[str, str] | None = None

        self.deriv_list_of_all_compounds: list[str] | None = None
        self.deriv_files_present: bool = False
        self.deriv_is_calibrations: dict[str:bool] = {}
        self.deriv_compounds_properties: pd.DataFrame | None = None
        self.deriv_is_files: dict[str, bool] | None = None

        self.samples_info: pd.DataFrame | None = None
        self.samples_info_std: pd.DataFrame | None = None
        self.samples: dict[str, pd.DataFrame] | None = None
        self.samples_std: dict[str, pd.DataFrame] | None = None

        self.list_of_files_param_reports = []
        self.list_of_files_param_aggrreps = []
        self.list_of_samples_param_reports = []
        self.list_of_samples_param_aggrreps = []
        self.files: dict[str, pd.DataFrame] = {}
        self.calibrations: dict[str : pd.DataFrame] = {}
        self.files_reports = {}
        self.files_aggrreps = {}
        self.samples_reports = {}
        self.samples_reports_std = {}
        self.samples_aggrreps = {}
        self.samples_aggrreps_std = {}

        self.columns_to_keep_in_files: list[str] = list(
            self.columns_to_rename_and_keep_in_files.keys()
        )
        self.acceptable_params: list[str] = list(self.param_to_axis_label.keys())

    def load_files_info(self, update_saved_files_info: bool = True) -> pd.DataFrame:
        """ """
        files_info_path = plib.Path(self.folder_path, "files_info.xlsx")
        if files_info_path.exists():
            files_info = pd.read_excel(
                files_info_path, engine="openpyxl", index_col="filename"
            )
            self.files_info = self._add_default_to_files_info(files_info)
            print("Info: files_info loaded")
        else:
            print("Info: files_info not found")
            self.files_info = self.create_files_info()
        if update_saved_files_info:
            self.files_info.to_excel(plib.Path(self.folder_path, "files_info.xlsx"))
        return self.files_info

    def create_files_info(self, update_saved_files_info: bool = False) -> pd.DataFrame:
        """ """
        filename: list[str] = [
            a.parts[-1].split(".")[0] for a in list(self.folder_path.glob("**/*.txt"))
        ]
        samplename = [f.split("_")[0] for f in filename]
        replicatenumber = [int(f.split("_")[1]) for f in filename]
        files_info_unsorted = pd.DataFrame(
            index=filename,
            data={
                "samplename": samplename,
                "replicatenumber": replicatenumber,
            },
        )
        files_info = files_info_unsorted.sort_index()
        files_info.index.name = "filename"
        self.files_info = self._add_default_to_files_info(files_info)
        if update_saved_files_info:
            self.files_info.to_excel(plib.Path(self.folder_path, "files_info.xlsx"))
        return self.files_info

    def _add_default_to_files_info(
        self, files_info_no_defaults: pd.DataFrame
    ) -> pd.DataFrame:
        """ """
        if "derivatized" not in list(files_info_no_defaults):
            files_info_no_defaults["derivatized"] = False
        if "calibration_file" not in list(files_info_no_defaults):
            files_info_no_defaults["calibration_file"] = False
        for col in self.files_info_defauls_columns:
            if col not in list(files_info_no_defaults):
                files_info_no_defaults[col] = 1
        return files_info_no_defaults

    def create_samples_info(self):
        """Creates a summary 'samples_info' DataFrame from 'files_info',
        aggregating data for each sample, and updates the 'samples_info'
        attribute with this summarized data."""
        if self.files_info is None:
            _ = self.load_files_info()
        self.samples_info = (
            self.files_info.reset_index().groupby("samplename").agg(list)
        )
        # self.samples_info.reset_index(inplace=True)
        self.samples_info.set_index("samplename", drop=True, inplace=True)
        print("Info: create_samples_info: samples_info created")
        return self.samples_info

    def load_all_files(self):
        """Loads all files listed in 'files_info' into a dictionary, where keys are
        filenames. Each file is processed to clean and standardize data. It updates the
        'files' attribute with data frames of file contents and 'is_files_deriv' with
        derivative information. Marks 'files_loaded' as True after loading."""
        print("Info: load_all_files: loop started")
        if self.files_info is None:
            self.load_files_info()
        for filename in self.files_info.index:
            file = self.load_single_file(filename)
            self.files[filename] = file
        print("Info: load_all_files: files loaded")
        return self.files

    def load_single_file(self, filename) -> pd.DataFrame:
        """Loads a single GCMS file by its name, cleans, and processes the data according
        to project settings (e.g., delimiter, columns to keep). It sums areas for duplicated
        compound names and handles dilution factors. Updates the file's data with iupac names
        and reorders columns. Logs the process and returns the cleaned DataFrame."""
        file = pd.read_csv(
            plib.Path(self.folder_path, filename + self.file_load_format),
            delimiter=self.file_load_delimiter,
            index_col=0,
            skiprows=self.file_load_skiprows,
        )
        columns_to_drop = [
            cl for cl in file.columns if cl not in self.columns_to_keep_in_files
        ]
        file.drop(columns_to_drop, axis=1, inplace=True)
        file.rename(
            self.columns_to_rename_and_keep_in_files, inplace=True, axis="columns"
        )

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
        file.rename(self.compounds_to_rename_in_files, inplace=True)
        print("\tInfo: load_single_file ", filename)
        return file

    def load_class_code_frac(self) -> pd.DataFrame:
        """ """
        class_code_frac_path = plib.Path(
            self.folder_path, "classifications_codes_fractions.xlsx"
        )
        if class_code_frac_path.exists():
            self.class_code_frac = pd.read_excel(class_code_frac_path)
        else:
            raise FileNotFoundError(
                '"classifications_codes_fractions.xlsx" not found in folder_path'
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
        if self.files_info is None:
            self.load_files_info()

        if any(self.files_info["calibration_file"]):
            for cal_name in set(self.files_info["calibration_file"].tolist()):
                cal_path = plib.Path(self.folder_path, cal_name + ".xlsx")
                if cal_path.exists():
                    cal_file = pd.read_excel(cal_path, index_col=0)
                else:
                    raise FileNotFoundError(f"{cal_name=} not found in folder_path")

                cal_file.index.name = "comp_name"
                cols_cal_area = [c for c in list(cal_file) if "Area" in c]
                cols_cal_ppms = [c for c in list(cal_file) if "PPM" in c]
                cal_file[cols_cal_area + cols_cal_ppms] = cal_file[
                    cols_cal_area + cols_cal_ppms
                ].apply(pd.to_numeric, errors="coerce")
                self.calibrations[cal_name] = cal_file
        return self.calibrations

    def create_list_of_all_compounds(self):
        """Compiles a list of all unique compounds across all loaded files and calibrations,
        only for underivatized compounds. It ensures all files
        are loaded before compiling the list, excludes 'unidentified' compounds, and updates
        the 'list_of_all_compounds' attribute. Logs completion and returns the list."""
        if not self.files:
            self.load_all_files()
        if not self.calibrations:
            self.load_calibrations()
        all_dfs_with_comps = [f for f in self.files.values()] + [
            f for f in self.calibrations.values()
        ]
        # non-derivatized compounds
        all_compounds: pd.DataFrame = pd.concat(all_dfs_with_comps)

        set_of_all_compounds = pd.Index(all_compounds.index.unique())
        # Using set comprehension to remove unwanted elements
        filtered_compounds = {
            compound.strip()  # Remove leading/trailing spaces
            for compound in set_of_all_compounds
            if compound not in ["unidentified", None, False, "", " ", "''"]
        }
        # Converting the filtered set to a list
        self.list_of_all_compounds = list(filtered_compounds)
        print(f"Info: created {len(self.list_of_all_compounds) = }")
        return self.list_of_all_compounds

    def create_compounds_properties(
        self, update_saved_files_info: bool = True
    ) -> pd.DataFrame:
        """Retrieves and organizes properties for underivatized compounds using pubchempy,
        updating the 'compounds_properties' attribute and saving the properties
        to 'compounds_properties.xlsx'."""
        print("Info: create_compounds_properties: started")

        if self.dict_classes_to_codes is None:
            self.load_class_code_frac()
        if self.list_of_all_compounds is None:
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
        # save db in the project folder in the input
        if update_saved_files_info:
            cpdf.to_excel(plib.Path(self.folder_path, "compounds_properties.xlsx"))
        print("Info: compounds_properties created")
        return self.compounds_properties

    def load_compounds_properties(self) -> pd.DataFrame:
        """Attempts to load the 'compounds_properties.xlsx' file containing physical
        and chemical properties of compounds. If not found, it creates a new properties
        DataFrame and updates the 'compounds_properties_created' attribute."""
        comps_prop_path = plib.Path(self.folder_path, "compounds_properties.xlsx")
        if comps_prop_path.exists():
            cpdf = pd.read_excel(comps_prop_path, index_col="comp_name")
            self.compounds_properties = cpdf
            print("Info: compounds_properties loaded")
        else:
            print("Warning: compounds_properties.xlsx not found")
            cpdf = self.create_compounds_properties()
        return self.compounds_properties

    def create_dict_names_to_iupacs(self) -> dict[str, str]:
        if self.compounds_properties is None:
            self.load_compounds_properties()
        self.dict_names_to_iupacs = self.compounds_properties["iupac_name"].to_dict()
        return self.dict_names_to_iupacs

    def add_iupac_to_files_and_calibrations(self):
        """Adds the IUPAC name to each compound in the loaded files,
        distinguishing between underivatized and derivatized compounds,
        and updates the corresponding file dataframes."""
        if not self.files:
            self.load_all_files()
        if self.compounds_properties is None:
            self.load_compounds_properties()
        for file in self.files.values():
            file["iupac_name"] = file.index.map(self.dict_names_to_iupacs)
        for file in self.calibrations.values():
            file["iupac_name"] = file.index.map(self.dict_names_to_iupacs)
        return self.files, self.calibrations

    # def apply_calibration_to_files(self):
    #     """Applies the appropriate calibration curve to each compound
    #     in the loaded files, adjusting concentrations based on calibration
    #     data, and updates the 'files' attribute with calibrated data."""
    #     print("Info: apply_calibration_to_files: loop started")
    #     if not self.files:
    #         self.load_all_files()
    #     if not self.calibrations:
    #         self.load_calibrations()
    #     if not self.iupac_to_files_added:
    #         _, _ = self.add_iupac_to_files()

    #     for filename, _ in self.files.items():
    #         calibration_name = self.files_info.loc[filename, "calibration_file"]
    #         calibration = self.calibrations[calibration_name]
    #         if not self.is_files_deriv[filename]:
    #             df_comps = self.compounds_properties
    #         else:
    #             df_comps = self.deriv_compounds_properties
    #         file = self._apply_calib_to_file(filename, calibration, df_comps)
    #         if Project.auto_save_to_excel:
    #             self.save_file(file, filename)
    #     self.calibration_to_files_applied = True
    #     return self.files, self.is_files_deriv


def create_tanimoto_matrix(smiles_list: list[str]):
    pass


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
