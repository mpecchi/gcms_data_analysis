from gcms_data_analysis.main import Project
import pathlib as plib
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

# example_data_path = plib.Path(plib.Path(__file__).parent.parent,
#     'example/data/')
example_data_path = r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\example\data"
Project.set_folder_path(example_data_path)
gcms = Project()
#%%
def print_checked_df_to_script_text(df):
    # Convert the DataFrame to a dictionary with 'list' orientation
    df_dict = df.to_dict(orient='list')

    # Convert the index to a list and get the index name
    index_list = df.index.tolist()
    index_name = df.index.name

    # Print the DataFrame reconstruction code with the index at the top
    print("pd.DataFrame(")
    # Print the index part first
    if index_name is not None:
        print(f"    index=pd.Index({index_list}, name='{index_name}'),")
    else:
        print(f"    index={index_list},")

    # Start printing the data dictionary
    print("    data={")
    # Print each column's data
    for key, values in df_dict.items():
        # Replace NaN values with np.nan for printing
        values_with_nan = [f"np.nan" if pd.isnull(value) else value for value in values]
        # Prepare the string representation of the list, handling np.nan specially
        values_str = str(values_with_nan).replace("'np.nan'", "np.nan")

        print(f"        '{key}': {values_str},")
    # Close the data dictionary
    print("    }")
    # Close the DataFrame construction
    print(")")
#%%