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
    "carbolic acid",  # same iupac but different comp_name
    "2,4,5-trichlorophenol",  # clorine (udentified)
    "phenoxytrimethylsilane",  # silane (not listed in fg)
    "bromophenol",  # Br not listed
    "9-octadecenoic acid, 1,2,3-propanetriyl ester, (e,e,e)-",  # large compound
    "name_not_on_pcp",  # test for legit string that gives no pcp result
    " ",  # wrong entry or datatype
    None,
    False,
    np.nan,
]

list_of_compound_properties: list[pd.DataFrame] = []
for compound in compounds:
    print(compound)
    n2p = name_to_properties(compound, dict_cl_to_codes, dict_cl_to_mass_fractions)
    list_of_compound_properties.append(n2p)

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

to_check.to_excel(
    plib.Path(
        r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\tests\data_name_to_properties",
        "checked_compounds_properties.xlsx",
    )
)
# %%
