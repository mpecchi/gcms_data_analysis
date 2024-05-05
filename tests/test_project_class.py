# %%
import pytest
import pathlib as plib
from gcms_data_analysis.gcms import Project
from pandas.testing import assert_frame_equal


folder_path: plib.Path = plib.Path(__file__).parent

folder_path = r"/Users/matteo/Projects/gcms_data_analysis/tests/data_minimal_case"
# %%
proj = Project(
    folder_path=folder_path,
    auto_save_to_excel=False,
    compounds_to_rename_in_files={"almost oleic acid": "oleic acid"},
)

# check a couple of defaults
assert proj.column_to_sort_values_in_samples == "retention_time"
assert proj.delta_mol_weight_threshold == 100
assert proj.acceptable_params == [
    "height",
    "area",
    "area_if_undiluted",
    "conc_vial_mg_L",
    "conc_vial_if_undiluted_mg_L",
    "fraction_of_sample_fr",
    "fraction_of_feedstock_fr",
]
# %%
fic = proj.create_files_info(update_saved_files_info=False)
fil = proj.load_files_info(update_saved_files_info=False)

fic.calibration_file = fil.calibration_file  # this cannot be updated automatically
assert_frame_equal(fil, fic, check_exact=False, atol=1e-5, rtol=1e-5)

# %%
files = proj.load_all_files()
# %%
ccf = proj.load_class_code_frac()

# %%
cal = proj.load_calibrations()
# %%
lac = proj.create_list_of_all_compounds()

# %%
cpc = proj.create_compounds_properties(update_saved_files_info=True)
cpl = proj.load_compounds_properties()
assert_frame_equal(cpc, cpl, check_exact=False, atol=1e-5, rtol=1e-5, check_dtype=False)
# %%
dni = proj.create_dict_names_to_iupacs()
assert "oleic acid" in dni.keys()
assert "notvalidcomp" in dni.keys()
assert "decanoic acid" in dni.values()
# %%
files_iupac, calibration_iupac = proj.add_iupac_to_files_and_calibrations()
# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import (
    GetMorganFingerprintAsBitVect,
)  # pylint: disable=no-name-in-module


def create_tanimoto_similarity_dict(
    comp_smiles: str, calib_smiless: list[str]
) -> dict[str, list[float]]:

    mols_comp = Chem.MolFromSmiles(comp_smiles)
    mols_cal = [
        Chem.MolFromSmiles(smi) for smi in calib_smiless if isinstance(smi, str)
    ]

    # Generate fingerprints from molecule objects, skipping None entries created from invalid SMILES
    fps_comp = GetMorganFingerprintAsBitVect(mols_comp, 2, nBits=1024)

    fps_cal = [
        GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        for mol in mols_cal
        if mol is not None
    ]

    # perform Tanimoto similarity betwenn the first and all other compounds
    similarity = DataStructs.BulkTanimotoSimilarity(fps_comp, fps_cal)
    # create a df with results
    return similarity


calib_comp_iupacs = proj.calibrations["cal_minimal"].index.tolist()
calib_smiless = proj.compounds_properties.loc[
    calib_comp_iupacs, "canonical_smiles"
].tolist()
tanimoto_similarity_df: pd.DataFrame = pd.DataFrame(
    index=proj.compounds_properties["iupac_name"],
    columns=calib_comp_iupacs,
)
for iupac, smiles in zip(
    proj.compounds_properties["iupac_name"],
    proj.compounds_properties["canonical_smiles"],
):
    if isinstance(smiles, str):
        sim = create_tanimoto_similarity_dict(smiles, calib_smiless)
        tanimoto_similarity_df.loc[iupac, :] = sim
# %%
compounds_properties = proj.compounds_properties
calib_comp_iupacs = proj.calibrations["cal_minimal"]["iupac_name"].tolist()
tanimoto_similarity_df: pd.DataFrame = pd.DataFrame(
    index=compounds_properties.iupac_name.tolist(),
    columns=proj.calibrations["cal_minimal"]["iupac_name"].tolist(),
)
for comp in compounds_properties.iupac_name.tolist():
    print(comp)
    s = create_tanimoto_similarity_dict(comp, calib_comp_iupacs)
    tanimoto_similarity_df.loc[comp, :] = s
# %%

# %%
