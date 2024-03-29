from gcms_data_analysis.main import Project
import pathlib as plib
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from collections.abc import Iterable

# example_data_path = plib.Path(plib.Path(__file__).parent.parent,
#     'example/data/')
example_data_path = r"C:\Users\mp933\OneDrive - Cornell University\Python\gcms_data_analysis\tests\data_for_testing"
Project.set_folder_path(example_data_path)
gcms = Project()


# %%
from collections.abc import Iterable


def print_checked_df_to_script_text_with_arrays(df):
    # Convert the DataFrame to a dictionary with 'list' orientation
    df_dict = df.to_dict(orient="list")

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
    # Iterate over each column and its values
    for key, values in df_dict.items():
        # Initialize a list to hold the processed values for each column
        processed_values = []
        for value in values:
            # Check if the value is an iterable (not a string) and convert to tuple
            if isinstance(value, Iterable) and not isinstance(value, str):
                processed_value = f"({', '.join(repr(v) for v in value)})"
            elif pd.isnull(value):
                # Handle NaN values
                processed_value = "np.nan"
            else:
                # Direct representation for other types
                processed_value = repr(value)
            processed_values.append(processed_value)

        # Join the processed values into a string representing a list or tuple
        values_str = f"[{', '.join(processed_values)}]"
        print(f"        '{key}': {values_str},")
    # Close the data dictionary and DataFrame construction
    print("    }")
    print(")")
