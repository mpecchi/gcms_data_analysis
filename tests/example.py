import pathlib as plib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import ele
import pubchempy as pcp
from gcms_data_analysis.fragmenter import Fragmenter

from gcms_data_analysis import name_to_properties, Project


folder_path = plib.Path(
    r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\tests\data_minimal_case"
)

Project.set_folder_path(folder_path)
Project.set_auto_save_to_excel(False)
gcms = Project()
# %%
to_check = gcms.create_files_info()


# %%
def test_load_files_info(gcms, checked_files_info):
    to_check = gcms.load_files_info()
    assert_frame_equal(
        to_check, checked_files_info, check_exact=False, atol=1e-5, rtol=1e-5
    )


# %%
to_check = gcms.create_files_info()
to_check = gcms.create_list_of_all_compounds()

# %%
