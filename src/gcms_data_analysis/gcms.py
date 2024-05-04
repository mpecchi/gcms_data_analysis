from __future__ import annotations
from typing import Literal
import pathlib as plib
import pandas as pd


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
        auto_save_reports: bool = True,
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
        self.auto_save_reports = auto_save_reports

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
        all_dfs_with_comps = []
        for file in self.files.values():
            all_dfs_with_comps.append(file)
        for calib in self.calibrations.values():
            all_dfs_with_comps.append(calib)
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

    def load_compounds_properties(self):
        """Attempts to load the 'compounds_properties.xlsx' file containing physical
        and chemical properties of compounds. If not found, it creates a new properties
        DataFrame and updates the 'compounds_properties_created' attribute."""
        compounds_properties_path = plib.Path(
            self.folder_path, "compounds_properties.xlsx"
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
