from setuptools import setup, find_packages


with open("README.md", 'r', encoding="utf-8") as f:
    description = f.read()

setup(
    name='gcms_data_analysis',  # Replace with your own package name
    version='0.1.4',  # Start with a small version number and increment it with each release
    author='mpecchi',  # Replace with your name
    description='Tool for automatic analysis of multiple HPLC results',  # Provide a short description
    long_description=description,  # This will read your README file to use as the long description
    long_description_content_type='text/markdown',  # This is the format of your README file
    url='https://github.com/mpecchi/gcms_data_analysis/tree/main',  # Replace with the URL of your project
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