from setuptools import setup, find_packages


# run the script_to_update_readme.py before
with open("README.md", 'r', encoding="utf-8") as f:
    readmecontent = f.read()

setup(
    name='gcms_data_analysis',  # Replace with your own package name
    version='1.0.2',  # Start with a small version number and increment it with each release
    author='Matteo Pecchi',  # Replace with your name
    description='Automatic analysis of GC-MS data',  # Provide a short description
    long_description=readmecontent,  # This will read your README file to use as the long description
    long_description_content_type='text/markdown',  # This is the format of your README file
    url='https://github.com/mpecchi/gcms_data_analysis/tree/main',  # Replace with the URL of your project
    project_urls={
        'Documentation': 'https://gcms-data-analysis.readthedocs.io/en/latest/',
    },
    packages=find_packages(),  # This function will find all the packages in your project
    install_requires=[
        'pathlib', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'ele', 'pubchempy',
        'rdkit', 'openpyxl', 'pyarrow'
    ],
    classifiers=[
        # Choose some classifiers from https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum version requirement of the Python for your package
)
