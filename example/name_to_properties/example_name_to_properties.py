import pathlib as plib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import ele
import pubchempy as pcp
from gcms_data_analysis.fragmenter import Fragmenter

from gcms_data_analysis import name_to_properties


# def get_compound_from_pubchempy(comp_name: str) -> pcp.Compound:
#     if not isinstance(comp_name, str):
#         return None
#     if comp_name == " " or comp_name == "":
#         return None
#     cond = True
#     while cond:  # to deal with HTML issues on server sides (timeouts)
#         try:
#             # comp contains all info about the chemical from pubchem
#             try:
#                 comp_inside_list = pcp.get_compounds(comp_name, "name")
#             except ValueError:
#                 print(f"{comp_name = }")
#                 return None
#             if comp_inside_list:
#                 comp = comp_inside_list[0]
#             else:
#                 print(
#                     f"WARNING: name_to_properties {comp_name=} does not find an entry in pcp",
#                 )
#                 return None
#             cond = False
#         except pcp.PubChemHTTPError:  # timeout error, simply try again
#             print("Caught: pcp.PubChemHTTPError (keep trying)")
#     return comp


# def _order_columns_in_compounds_properties(
#     unsorted_df: pd.DataFrame | None,
# ) -> pd.DataFrame | None:
#     if unsorted_df is None:
#         return None

#     # Define a custom sort key function
#     def sort_key(col):
#         if col.startswith("el_mf"):
#             return (2, col)
#         elif col.startswith("el_"):
#             return (1, col)
#         elif col.startswith("fg_mf_unclassified"):
#             return (5, col)
#         elif col.startswith("fg_mf"):
#             return (4, col)
#         elif col.startswith("fg_"):
#             return (3, col)
#         else:
#             return (0, col)

#     # Sort columns using the custom key
#     sorted_columns = sorted(unsorted_df.columns, key=sort_key)
#     sorted_df = unsorted_df.reindex(sorted_columns, axis=1)
#     sorted_df.index.name = "comp_name"
#     # Reindex the DataFrame with the sorted columns
#     return sorted_df


# def name_to_properties2(
#     comp_name: str,
#     dict_classes_to_codes: dict[str:str],
#     dict_classes_to_mass_fractions: dict[str:float],
#     df: pd.DataFrame | None = None,
#     precision_sum_elements: float = 0.05,
#     precision_sum_functional_group: float = 0.05,
# ) -> pd.DataFrame | None:
#     """
#     used to retrieve chemical properties of the compound indicated by the
#     comp_name and to store those properties in the df

#     Parameters
#     ----------
#     GCname : str
#         name from GC, used as a unique key.
#     search_name : str
#         name to be used to search on pubchem.
#     df : pd.DataFrame
#         that contains all searched compounds.
#     df_class_code_frac : pd.DataFrame
#         contains the list of functional group names, codes to be searched
#         and the weight fraction of each one to automatically calculate the
#         mass fraction of each compounds for each functional group.
#         Classes are given as smarts and are looked into the smiles of the comp.

#     Returns
#     -------
#     df : pd.DataFrame
#         updated dataframe with the searched compound.
#     CompNotFound : str
#         if GCname did not yield anything CompNotFound=GCname.

#     """
#     # classes used to split compounds into functional groups
#     comp = get_compound_from_pubchempy(comp_name)

#     if comp is None:
#         if not isinstance(comp_name, str):
#             return df
#         else:
#             if not comp_name or comp_name.isspace():
#                 return df
#             else:
#                 if df is not None:
#                     df.loc[comp_name, "iupac_name"] = "unidentified"
#                 return df
#     if df is None:
#         df = pd.DataFrame(dtype=float)
#     try:
#         df.loc[comp_name, "iupac_name"] = comp.iupac_name.lower()
#     except AttributeError:  # iupac_name not give
#         df.loc[comp_name, "iupac_name"] = comp_name.lower()
#     df.loc[comp_name, "molecular_formula"] = comp.molecular_formula
#     df.loc[comp_name, "canonical_smiles"] = comp.canonical_smiles
#     df.loc[comp_name, "molecular_weight"] = float(comp.molecular_weight)

#     try:
#         df.loc[comp_name, "xlogp"] = float(comp.xlogp)
#     except (
#         TypeError
#     ):  # float() argument must be a string or a real number, not 'NoneType'
#         df.loc[comp_name, "xlogp"] = np.nan
#     elements = set(comp.to_dict()["elements"])
#     el_dict = {}
#     el_mf_dict = {}

#     for el in elements:
#         el_count = comp.to_dict()["elements"].count(el)
#         el_mass = ele.element_from_symbol(el).mass

#         # Using similar logic as in the fg_dict example
#         if el not in el_dict:
#             el_dict[el] = 0
#             el_mf_dict[el] = 0.0

#         el_dict[el] += int(el_count)
#         el_mf_dict[el] += (
#             float(el_count) * float(el_mass) / float(comp.molecular_weight)
#         )
#     # Now, update the DataFrame in a similar way to the fg_dict example
#     for key, value in el_dict.items():
#         df.at[comp_name, f"el_{key}"] = int(value)

#     for key, value in el_mf_dict.items():
#         df.at[comp_name, f"el_{key}"] = float(value)
#     cols_el_mf = [col for col in df.columns if col.startswith("el_mf")]
#     residual_els = df.loc[comp_name, cols_el_mf].sum() - 1
#     # check element sum
#     try:
#         assert residual_els <= precision_sum_elements
#     except AssertionError:
#         raise AssertionError(
#             f"the total mass fraction of elements in {comp_name =} is > 0.001"
#         )
#     # apply fragmentation using the Fragmenter class (thanks simonmb)
#     frg = Fragmenter(
#         dict_classes_to_codes,
#         fragmentation_scheme_order=dict_classes_to_codes.keys(),
#         algorithm="simple",
#     )
#     fragmentation, _, _ = frg.fragment(comp.canonical_smiles)
#     fg_dict = {}
#     fg_mf_dict = {}
#     # Iterate over each item in the dictionary
#     for key, value in fragmentation.items():
#         # Determine the root key (the part before an underscore, if present)
#         root_key = key.split("_")[0]
#         # if root_key in hetero_atoms:
#         #     pass
#         # Check if the root key is in the sum_dict; if not, initialize it
#         if root_key not in fg_dict:
#             fg_dict[root_key] = 0
#             fg_mf_dict[root_key] = 0
#         # Add the value to the corresponding root key in the sum_dict
#         fg_dict[root_key] += int(fragmentation[key])
#         fg_mf_dict[root_key] += (
#             float(fragmentation[key])
#             * float(dict_classes_to_mass_fractions[key])
#             / df.loc[comp_name, "molecular_weight"].astype(float)
#         )  # mass fraction of total

#     # Update df with fg_dict
#     for key, value in fg_dict.items():
#         df.at[comp_name, f"fg_{key}"] = int(value)  # Update the cell
#     # Update df with fg_mf_dict
#     for key, value in fg_mf_dict.items():
#         df.at[comp_name, f"fg_mf_{key}"] = float(value)  # Update the cell
#     cols_fg_mf = [col for col in df.columns if col.startswith("fg_mf")]
#     residual_fgs = df.loc[comp_name, cols_fg_mf].sum() - 1
#     try:
#         assert residual_fgs <= precision_sum_functional_group
#     except AssertionError:
#         print(f"{df.loc[comp_name, cols_fg_mf].sum()=}")
#         raise AssertionError(
#             f"the total mass fraction of functional groups in {comp_name =} is > 0.05"
#         )
#     if residual_fgs < -precision_sum_functional_group:
#         df.at[comp_name, f"fg_mf_unclassified"] = abs(residual_fgs)
#     df.loc[df["iupac_name"] != "unidentified"] = df.loc[
#         df["iupac_name"] != "unidentified"
#     ].fillna(0)
#     df = _order_columns_in_compounds_properties(df)

#     return df


folder_path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\tests\data_name_to_properties"
)
# %%
classifications_codes_fractions = pd.read_excel(
    plib.Path(
        folder_path,
        "classifications_codes_fractions.xlsx",
    )
)
checked_compounds_properties = pd.read_excel(
    plib.Path(
        folder_path,
        "checked_compounds_properties.xlsx",
    ),
    index_col="comp_name",
)
dict_cl_to_codes: dict[str, str] = dict(
    zip(
        classifications_codes_fractions.classes.tolist(),
        classifications_codes_fractions.codes.tolist(),
    )
)
dict_cl_to_mass_fractions: dict[str, float] = dict(
    zip(
        classifications_codes_fractions.classes.tolist(),
        classifications_codes_fractions.mfs.tolist(),
    )
)
# %%

compounds = [
    "2-methylcyclopent-2-en-1-one",  # small ketone
    "hexadecanoic acid",
    "n-hexadecanoic acid",  # different names same compounds
    "phenol",  # ring
    "phenol",  # repeated compound
    "2,4,5-trichlorophenol",  # clorine (udentified)
    "phenoxytrimethylsilane",  # silane (not listed in fg)
    "bromophenol",  # Br not listed
    "9-octadecenoic acid, 1,2,3-propanetriyl ester, (e,e,e)-",  # large compound
    "wrong_name",  # test for legit string that gives no pcp result
    " ",  # wrong entry or datatype
    None,
    False,
    np.nan,
]

list_of_compound_properties: list[pd.DataFrame] = []
for compound in compounds:
    print(compound)
    n2p = name_to_properties(
        compound, dict_cl_to_codes, dict_cl_to_mass_fractions, None
    )
    list_of_compound_properties.append(n2p)
    if n2p is not None:
        to_check = n2p.loc[[compound], :]
        to_check = to_check.loc[:, (to_check != 0).any(axis=0)]
        checked = checked_compounds_properties.loc[[compound], :]
        checked = checked.loc[:, (checked != 0).any(axis=0)]
        pd.testing.assert_frame_equal(
            to_check,
            checked,
            check_exact=False,
            atol=1e-5,
            rtol=1e-5,
        )
# %%
to_check = pd.DataFrame()
for compound in compounds:
    print(compound)
    to_check = name_to_properties(
        compound,
        dict_cl_to_codes,
        dict_cl_to_mass_fractions,
        to_check,
    )
pd.testing.assert_frame_equal(
    to_check,
    checked_compounds_properties,
    check_exact=False,
    atol=1e-5,
    rtol=1e-5,
)

# %%
