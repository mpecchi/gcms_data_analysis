from __future__ import annotations
from typing import Literal
import pathlib as plib
import numpy as np
import pandas as pd
import pubchempy as pcp
import ele
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import (  # pylint: disable=no-name-in-module
    GetMorganFingerprintAsBitVect,
)
from gcms_data_analysis.fragmenter import Fragmenter


class Project:
    """
    Represents a project (identified by the folder where the data is stored)
    for TGA data analysis.

    :param folder_path: The path to the folder where the data is stored.
    :type folder_path: Path or str
    :param name: The name of the project. If not provided, it will be inferred from the folder path.
    :type name: str, optional
    :param use_semi_calibration: Whether to use semi-calibration for data analysis. Defaults to True.
    :type use_semi_calibration: bool, optional
    :param tanimoto_similarity_threshold: The threshold for Tanimoto similarity. Defaults to 0.4.
    :type tanimoto_similarity_threshold: float, optional
    :param delta_mol_weight_threshold: The threshold for delta molecular weight. Defaults to 100.
    :type delta_mol_weight_threshold: int, optional
    :param file_load_skiprows: The number of rows to skip when loading files. Defaults to 8.
    :type file_load_skiprows: int, optional
    :param file_load_delimiter: The delimiter used in the files. Defaults to "\t".
    :type file_load_delimiter: {"\t", ",", ";"}, optional
    :param file_load_format: The format of the files to load. Defaults to ".txt".
    :type file_load_format: {".txt", ".csv"}, optional
    :param column_to_sort_values_in_samples: The column to sort values in samples. Defaults to "retention_time".
    :type column_to_sort_values_in_samples: {"retention_time", "area", "height"}, optional
    :param plot_font: The font to use in plots. Defaults to "Dejavu Sans".
    :type plot_font: {"Dejavu Sans", "Times New Roman"}, optional
    :param plot_grid: Whether to show grid lines in plots. Defaults to False.
    :type plot_grid: bool, optional
    :param auto_save_to_excel: Whether to automatically save data to Excel files. Defaults to True.
    :type auto_save_to_excel: bool, optional
    :param columns_to_rename_and_keep_in_files: A dictionary mapping column names to new names to rename and keep in files.
        If not provided, default mappings will be used. Defaults to None.
    :type columns_to_rename_and_keep_in_files: dict[str, str] or None, optional
    :param compounds_to_rename_in_files: A dictionary mapping compound names to new names to rename in files.
        If not provided, no renaming will be performed. Defaults to None.
    :type compounds_to_rename_in_files: dict[str, str] or None, optional
    :param param_to_axis_label: A dictionary mapping parameter names to axis labels.
        If not provided, default mappings will be used. Defaults to None.
    :type param_to_axis_label: dict[str, str] or None, optional
    :param string_in_deriv_names: A list of strings that may appear in derivative names.
        If not provided, default strings will be used. Defaults to None.
    :type string_in_deriv_names: list[str] or None, optional
    """

    def __init__(
        self,
        folder_path: plib.Path | str,
        name: str | None = None,
        use_semi_calibration: bool = True,
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
        self.use_semi_calibration = use_semi_calibration
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

        self.files: dict[str, pd.DataFrame] = {}
        self.samples: dict[str, pd.DataFrame] = {}
        self.samples_std: dict[str, pd.DataFrame] = {}
        self.calibrations: dict[str : pd.DataFrame] = {}
        self.tanimoto_similarity_df: dict[str : pd.DataFrame] = {}
        self.molecular_weight_diff_df: dict[str : pd.DataFrame] = {}
        self.semi_calibration_dict: dict[str, dict[str, str]] = {}
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
        """
        Loads the files information from an Excel file and returns it as a DataFrame.

        :param update_saved_files_info: Specifies whether to update the saved files_info.xlsx file.
        :type update_saved_files_info: bool, optional
        :return: The loaded files information as a DataFrame.
        :rtype: pd.DataFrame
        """

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
        """
        Create a DataFrame containing information about the files in the folder.

        :param update_saved_files_info: Whether to update the saved files_info.xlsx file. Defaults to False.
        :type update_saved_files_info: bool, optional
        :return: The DataFrame containing the files information.
        :rtype: pd.DataFrame
        """
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
        """Add default values to the files_info DataFrame.

        This method takes a DataFrame `files_info_no_defaults` as input and adds default values to it.
        The default values are added for the columns 'derivatized', 'calibration_file',
        and any other columns specified in `self.files_info_defauls_columns`.

        Args:
            files_info_no_defaults (pd.DataFrame): The DataFrame containing files_info without default values.

        Returns:
            pd.DataFrame: The DataFrame with default values added.

        """
        if "derivatized" not in list(files_info_no_defaults):
            files_info_no_defaults["derivatized"] = False
        if "calibration_file" not in list(files_info_no_defaults):
            files_info_no_defaults["calibration_file"] = False
        for col in self.files_info_defauls_columns:
            if col not in list(files_info_no_defaults):
                files_info_no_defaults[col] = 1
        return files_info_no_defaults

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
        if not self.calibrations:
            self.load_calibrations()
        if self.compounds_properties is None:
            self.load_compounds_properties()
        if self.dict_names_to_iupacs is None:
            self.create_dict_names_to_iupacs()
        for file in self.files.values():
            file["iupac_name"] = file.index.map(self.dict_names_to_iupacs)
        for cal in self.calibrations.values():
            cal["iupac_name"] = cal.index.map(self.dict_names_to_iupacs)
        return self.files, self.calibrations

    def create_tanimoto_and_molecular_weight_similarity_dfs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.files:
            self.load_all_files()
        if not self.calibrations:
            self.load_calibrations()
        if self.compounds_properties is None:
            self.load_compounds_properties()
        if self.dict_names_to_iupacs is None:
            self.create_dict_names_to_iupacs()
        if "iupac_name" not in list(self.files.values())[0].columns:
            self.add_iupac_to_files_and_calibrations()
        prop_index_iupac = self.compounds_properties.set_index("iupac_name")
        prop_index_iupac = prop_index_iupac[
            ~prop_index_iupac.index.duplicated(keep="first")
        ].dropna(how="all", axis=0)
        for calibrationname, calibration in self.calibrations.items():
            calib_iupacs = calibration["iupac_name"].tolist()
            calib_smiless = prop_index_iupac.loc[
                calib_iupacs, "canonical_smiles"
            ].tolist()
            calib_mws = prop_index_iupac.loc[
                calib_iupacs, "molecular_weight"
            ].to_numpy()
            non_calib_iupacs = [
                p for p in prop_index_iupac.index if p not in calib_iupacs
            ]
            non_calib_smiless = prop_index_iupac.loc[
                non_calib_iupacs, "canonical_smiles"
            ].tolist()
            non_calib_mws = prop_index_iupac.loc[
                non_calib_iupacs, "molecular_weight"
            ].to_numpy()

            tan_sim_df = pd.DataFrame(index=non_calib_iupacs, columns=calib_iupacs)
            mw_diff_df = pd.DataFrame(index=non_calib_iupacs, columns=calib_iupacs)
            for iupac, smiles, weight in zip(
                non_calib_iupacs, non_calib_smiless, non_calib_mws
            ):
                if isinstance(smiles, str):
                    tan_sim_df.loc[iupac, :] = create_tanimoto_similarity_dict(
                        smiles, calib_smiless
                    )
                    mw_diff_df.loc[iupac, :] = np.abs(calib_mws - weight)
            self.tanimoto_similarity_df[calibrationname] = tan_sim_df
            self.molecular_weight_diff_df[calibrationname] = mw_diff_df
            return (
                self.tanimoto_similarity_df[calibrationname],
                self.molecular_weight_diff_df[calibrationname],
            )

    def create_semi_calibration_dict(self) -> dict[str, dict[str, str]]:
        if not self.tanimoto_similarity_df or not self.molecular_weight_diff_df:
            self.create_tanimoto_and_molecular_weight_similarity_dfs()
        for calibrationname in self.calibrations.keys():
            if self.tanimoto_similarity_threshold is not None:
                all_valid_ts = self.tanimoto_similarity_df[calibrationname].where(
                    self.tanimoto_similarity_df[calibrationname]
                    >= self.tanimoto_similarity_threshold
                )
            else:
                all_valid_ts = self.tanimoto_similarity_df[calibrationname]
            all_valid_ts.dropna(axis=0, how="all", inplace=True)
            # Identify the column with the highest value in each row that meets the threshold
            best_valid_ts = all_valid_ts.idxmax(axis=1)

            if self.delta_mol_weight_threshold is not None:
                all_valid_mw = self.molecular_weight_diff_df[calibrationname].where(
                    self.molecular_weight_diff_df[calibrationname]
                    <= self.delta_mol_weight_threshold
                )
            else:
                all_valid_mw = self.molecular_weight_diff_df[calibrationname]
            all_valid_mw.dropna(axis=0, how="all", inplace=True)
            # Identify the column with the highest value in each row that meets the threshold
            best_valid_mw = all_valid_mw.idxmin(axis=1)
            self.semi_calibration_dict[calibrationname] = {
                k: best_valid_ts[k]
                for k in best_valid_ts.keys()
                if k in best_valid_mw and best_valid_ts[k] == best_valid_mw[k]
            }
            return self.semi_calibration_dict[calibrationname]

    def apply_calibration_to_files(self):
        """Applies the appropriate calibration curve to each compound
        in the loaded files, adjusting concentrations based on calibration
        data, and updates the 'files' attribute with calibrated data."""
        print("Info: apply_calibration_to_files: loop started")
        if "iupac_name" not in list(self.files.values())[0].columns:
            self.add_iupac_to_files_and_calibrations()
        if self.use_semi_calibration and not self.semi_calibration_dict:
            self.create_semi_calibration_dict()

        for filename in self.files.keys():
            self.files[filename] = self.apply_calib_to_single_file(filename)
        return self.files

    def apply_calib_to_single_file(self, filename) -> pd.DataFrame:
        """computes conc data based on the calibration provided.
        If semi_calibration is specified, the closest compound in terms of
        Tanimoto similarity and molecular weight similarity is used for
        compounds where a calibration entry is not available"""
        # """calibration.rename(Project.compounds_to_rename, inplace=True)"""
        # print(file)
        print("\tInfo: _apply_calib_to_file ", filename)
        calibrationname = self.files_info.loc[filename, "calibration_file"]
        clbrtn = self.calibrations[calibrationname].set_index("iupac_name")
        if self.use_semi_calibration:
            semi_cal_dic = self.semi_calibration_dict[calibrationname]
        cols_cal_area = [c for c in list(clbrtn) if "Area" in c]
        cols_cal_ppms = [c for c in list(clbrtn) if "PPM" in c]
        tot_sample_conc = self.files_info.loc[
            filename, "total_sample_conc_in_vial_mg_L"
        ]
        sample_yield_feed_basis = self.files_info.loc[
            filename, "sample_yield_on_feedstock_basis_fr"
        ]

        for compname, compiupac in zip(
            self.files[filename].index.tolist(),
            self.files[filename]["iupac_name"].tolist(),
        ):
            if compiupac == "unidentified":
                iupac_for_calib = "n.a."
            else:
                if compiupac in clbrtn.index.tolist():
                    iupac_for_calib = compiupac
                else:
                    if self.use_semi_calibration:
                        if compiupac in list(semi_cal_dic.keys()):
                            iupac_for_calib = semi_cal_dic[compiupac]
                        else:
                            iupac_for_calib = "n.a."
            if iupac_for_calib != "n.a.":
                # areas and ppms for the calibration are taken from df_clbr
                cal_areas = clbrtn.loc[iupac_for_calib, cols_cal_area].to_numpy(
                    dtype=float
                )
                cal_ppms = clbrtn.loc[iupac_for_calib, cols_cal_ppms].to_numpy(
                    dtype=float
                )
                # linear fit of calibration curve (exclude nan), get ppm from area
                fit = np.polyfit(
                    cal_areas[~np.isnan(cal_areas)], cal_ppms[~np.isnan(cal_ppms)], 1
                )
                # concentration at the injection solution (GC vial) ppp = mg/L
                conc_mg_l = np.poly1d(fit)(self.files[filename].loc[compname, "area"])
                if conc_mg_l < 0:
                    conc_mg_l = 0
            else:
                conc_mg_l = np.nan
            self.files[filename].loc[compname, "conc_vial_mg_L"] = conc_mg_l
            self.files[filename].loc[compname, "conc_vial_if_undiluted_mg_L"] = (
                conc_mg_l * self.files_info.loc[filename, "dilution_factor"]
            )
            self.files[filename].loc[compname, "fraction_of_sample_fr"] = (
                conc_mg_l / tot_sample_conc
            )
            self.files[filename].loc[compname, "fraction_of_feedstock_fr"] = (
                conc_mg_l / tot_sample_conc * sample_yield_feed_basis
            )
            self.files[filename].loc[compname, "calibration_used"] = iupac_for_calib
        if np.isnan(self.files[filename]["conc_vial_mg_L"]).all():
            print(
                f"WARNING: the file {filename} does not contain any ",
                "compound for which a calibration nor a semicalibration is available.",
                "\n either lower similarity thresholds, add calibration compounds, or",
                "calibration_file=False in files_info.xlsx",
            )
        return self.files[filename]

    def add_stats_to_files_info(self) -> pd.DataFrame:
        """Computes and adds statistical data for each file to the 'files_info'
        DataFrame, such as maximum height, area, and concentrations,
        updating the 'files_info' with these statistics."""
        print("Info: add_stats_to_files_info: started")

        numeric_columns = [
            col
            for col in self.acceptable_params
            if col in list(self.files.values())[0].columns
        ]
        max_columns = [f"max_{nc}" for nc in numeric_columns]
        total_columns = [f"total_{nc}" for nc in numeric_columns]
        comp_with_max_columns = [f"compound_with_max_{nc}" for nc in numeric_columns]
        for name, df in self.files.items():
            for ncol, mcol, tcol, cmcol in zip(
                numeric_columns, max_columns, total_columns, comp_with_max_columns
            ):
                self.files_info.loc[name, mcol] = df[ncol].max()
                self.files_info.loc[name, tcol] = df[ncol].sum()
                self.files_info.loc[name, cmcol] = df[df[ncol] == df[ncol].max()].index[
                    0
                ]
        # convert max and total columns to float
        for col in max_columns + total_columns:
            if col in self.files_info.columns:
                self.files_info[col] = self.files_info[col].astype(float)
        return self.files_info

    def create_samples_info(self):
        """Creates a summary 'samples_info' DataFrame from 'files_info',
        aggregating data for each sample, and updates the 'samples_info'
        attribute with this summarized data."""
        if self.files_info is None:
            self.load_files_info()
        numeric_columns = [
            col
            for col in self.acceptable_params
            if col in list(self.files.values())[0].columns
        ]
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
        print("Info: create_samples_info: samples_info created")
        return self.samples_info, self.samples_info_std

    def create_single_sample_from_files(
        self, files_in_sample: list[pd.DataFrame], samplename: str
    ):
        """Creates a sample dataframe and a standard deviation dataframe from files
        that are replicates of the same sample. This process includes aligning dataframes,
        filling missing values, calculating averages and standard deviations,
        and merging non-numerical data.

        :param files_in_sample: A list of pandas DataFrames representing replicates of the same sample.
        :param samplename: The name of the sample.

        :return: A tuple containing the sample dataframe and the standard deviation dataframe.
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        """
        all_ordered_columns = files_in_sample[0].columns.tolist()

        non_num_columns = [
            col
            for col in ["iupac_name", "calibration_used"]
            if col in all_ordered_columns
        ]
        # Step 1: Create a comprehensive index of all unique compounds
        all_compounds = pd.Index([])
        for df in files_in_sample:
            all_compounds = all_compounds.union(df.index)

        # Step 2: Align all DataFrames to the comprehensive index
        aligned_dfs: list[pd.DataFrame] = [
            df.reindex(all_compounds) for df in files_in_sample
        ]
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
        sample = sample.sort_values(by=self.column_to_sort_values_in_samples)
        # Apply the same order to 'sample_std' using reindex
        sample_std = sample_std.reindex(sample.index)
        sample = sample[all_ordered_columns]
        sample_std = sample_std[all_ordered_columns]
        sample.index.name = samplename
        sample_std.index.name = samplename

        return sample, sample_std

    def create_samples_from_files(self):
        """Generates a DataFrame for each sample by averaging and calculating
        the standard deviation of replicates, creating a comprehensive
        dataset for each sample in the project.

        Returns:
            tuple: A tuple containing two dictionaries. The first dictionary
            contains the generated DataFrame for each sample, where the key
            is the sample name and the value is the DataFrame. The second
            dictionary contains the standard deviation for each sample, where
            the key is the sample name and the value is the standard deviation.
        """
        if self.samples_info is None:
            self.create_samples_info()
        for samplename in self.samples_info.index:
            print("Sample: ", samplename)
            _files = []
            for filename in self.files_info.index[
                self.files_info["samplename"] == samplename
            ]:
                print("\tFile: ", filename)
                _files.append(self.files[filename])
            sample, sample_std = self.create_single_sample_from_files(
                _files, samplename
            )
            self.samples[samplename] = sample
            self.samples_std[samplename] = sample_std
        return self.samples, self.samples_std

    def create_files_param_report(self, param="conc_vial_mg_L"):
        """
        Create a report that consolidates the values of a specified parameter from different DataFrames,
        using the union of all indices found in the individual DataFrames.

        :param param: The parameter to extract from each DataFrame. Defaults to "conc_vial_mg_L".
        :return: A DataFrame containing the consolidated report.
        """
        if not self.files:
            self.load_all_files()
        if param not in self.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        # Create a dictionary of Series, each Series named after the file and containing the 'param' values
        series_dict = {
            filename: self.files[filename][param].rename(filename)
            for filename in self.files_info.index
        }
        # Get the union of all indices from the individual DataFrames
        rep = pd.concat(
            series_dict.values(), axis=1, keys=series_dict.keys(), join="outer"
        )
        # Reindex the DataFrame to include all unique indices, filling missing values with 0
        rep = rep.fillna(0)
        rep = rep.sort_index(key=rep.max(axis=1).get, ascending=False)
        # remove null columns from rep
        rep = rep.loc[rep.any(axis=1), :]
        # Save and return the report
        self.files_reports[param] = rep
        return self.files_reports[param]

    def create_files_param_aggrrep(self, param="conc_vial_mg_L"):
        """Aggregates compound concentration data by functional group for each
        parameter across all FILES, providing a summarized view of functional
        group concentrations. This aggregation facilitates the understanding
        of functional group distribution across FILES."""
        print("Info: create_param_aggrrep: ", param)
        if param not in self.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.files_reports:
            self.create_files_param_report(param)
        # create a df with iupac name index and fg_mf columns (underiv and deriv)
        comps_df = self.compounds_properties  # .set_index("iupac_name")
        # comps_df = comps_df[~comps_df.index.duplicated(keep="first")]

        # fg = functional groups, mf = mass fraction
        filenames = self.files_info.index.tolist()
        _all_comps = self.files_reports[param].index.tolist()
        _all_comps = [comp for comp in _all_comps if comp != "unidentified"]
        fg_mf_labs = [
            c for c in comps_df.columns if c.startswith("fg_mf_") if c != "fg_mf_total"
        ]
        fg_labs = [c[6:] for c in fg_mf_labs]

        fg_mf_all = pd.DataFrame(index=_all_comps, columns=fg_mf_labs)
        for idx in fg_mf_all.index.tolist():
            fg_mf_all.loc[idx, fg_mf_labs] = comps_df.loc[idx, fg_mf_labs]
        # create the aggregated dataframes and compute aggregated results
        aggrrep = pd.DataFrame(columns=filenames, index=fg_labs, dtype="float")
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
        aggrrep = aggrrep.loc[aggrrep.any(axis=1), :]  # drop rows with only 0
        aggrrep = aggrrep.sort_index(key=aggrrep[filenames].max(1).get, ascending=False)
        self.files_aggrreps[param] = aggrrep
        return aggrrep

    def create_samples_param_report(self, param: str = "conc_vial_mg_L"):
        print(f"Info: create_samples_param_report: {param = }")
        if param not in self.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.files_reports:
            self.create_files_param_report(param)
        file_to_sample_rename = dict(
            zip(self.files_info.index.tolist(), self.files_info["samplename"])
        )
        filerep = self.files_reports[param].copy()
        filerep.rename(columns=file_to_sample_rename, inplace=True)
        self.samples_reports[param] = filerep.T.groupby(by=filerep.columns).mean().T
        self.samples_reports_std[param] = filerep.T.groupby(by=filerep.columns).std().T
        return self.samples_reports[param], self.samples_reports_std[param]

    def create_samples_param_aggrrep(self, param: str = "conc_vial_mg_L"):
        print(f"Info: create_samples_param_aggrrep: {param = }")
        if param not in self.acceptable_params:
            raise ValueError(f"{param = } is not an acceptable param")
        if param not in self.files_aggrreps:
            self.create_files_param_aggrrep(param)
        file_to_sample_rename = dict(
            zip(self.files_info.index.tolist(), self.files_info["samplename"])
        )
        fileagg = self.files_aggrreps[param].copy()
        fileagg.rename(columns=file_to_sample_rename, inplace=True)
        self.samples_aggrreps[param] = fileagg.T.groupby(by=fileagg.columns).mean().T
        self.samples_aggrreps_std[param] = fileagg.T.groupby(by=fileagg.columns).std().T
        return self.samples_aggrreps[param], self.samples_aggrreps_std[param]

    def save_files_samples_reports(self):
        """"""
        for subfolder in [
            "",
            "files",
            "samples",
            "files_reports",
            "files_aggrreps",
            "samples_reports",
            "samples_aggrreps",
        ]:
            plib.Path(self.out_path, subfolder).mkdir(parents=True, exist_ok=True)
        out_path = self.out_path
        # save files_info and samples_info to the general output folder
        if self.files_info is not None:
            self.files_info.to_excel(plib.Path(out_path, "files_info.xlsx"))
        if self.samples_info is not None:
            self.samples_info.to_excel(plib.Path(out_path, "samples_info.xlsx"))
            self.samples_info_std.to_excel(plib.Path(out_path, "samples_info_std.xlsx"))
        if self.files:
            for filename, df in self.files.items():
                df.to_excel(plib.Path(out_path, "files", f"{filename}.xlsx"))
        if self.samples:
            for samplename, df in self.samples.items():
                df.to_excel(plib.Path(out_path, "samples", f"{samplename}.xlsx"))
            for samplename, df in self.samples_std.items():
                df.to_excel(plib.Path(out_path, "samples", f"{samplename}_std.xlsx"))
        if self.files_reports:
            for param, df in self.files_reports.items():
                df.to_excel(
                    plib.Path(out_path, "files_reports", f"report_files_{param}.xlsx")
                )
        if self.files_aggrreps:
            for param, df in self.files_aggrreps.items():
                df.to_excel(
                    plib.Path(
                        self.out_path, "files_aggrreps", f"aggrrep_files_{param}.xlsx"
                    )
                )
        if self.samples_reports:
            for param, df in self.samples_reports.items():
                df.to_excel(
                    plib.Path(
                        self.out_path, "samples_reports", f"report_samples_{param}.xlsx"
                    )
                )
            for param, df in self.samples_reports_std.items():
                df.to_excel(
                    plib.Path(
                        self.out_path,
                        "samples_reports",
                        f"report_samples_{param}_std.xlsx",
                    )
                )
        if self.samples_aggrreps:
            for param, df in self.samples_aggrreps.items():
                df.to_excel(
                    plib.Path(
                        self.out_path,
                        "samples_aggrreps",
                        f"aggrrep_samples_{param}.xlsx",
                    )
                )
            for param, df in self.samples_aggrreps_std.items():
                df.to_excel(
                    plib.Path(
                        self.out_path,
                        "samples_aggrreps",
                        f"aggrrep_samples_{param}_std.xlsx",
                    )
                )


def create_tanimoto_similarity_dict(
    comp_smiles: str, calib_smiless: list[str]
) -> dict[str, list[float]]:

    mols_comp = Chem.MolFromSmiles(comp_smiles)  # pylint: disable=no-member
    mols_cal = [
        Chem.MolFromSmiles(smi)  # pylint: disable=no-member
        for smi in calib_smiless
        if isinstance(smi, str)
    ]

    # Generate fingerprints from molecule objects, skipping None entries created from invalid SMILES
    fps_comp = GetMorganFingerprintAsBitVect(mols_comp, 2, nBits=1024)

    fps_cal = [
        GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for mol in mols_cal
        if mol is not None
    ]

    # perform Tanimoto similarity betwenn the first and all other compounds
    similarity = DataStructs.BulkTanimotoSimilarity(  # pylint: disable=no-member
        fps_comp, fps_cal
    )
    # create a df with results
    return similarity


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
