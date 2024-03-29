# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

project = 'gcms_data_analysis'
copyright = '2024, Matteo Pecchi'
author = 'Matteo Pecchi'
release = '1.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',  # Include autodoc extension for automatic documentation generation
    'sphinx.ext.viewcode',  # Include viewcode extension for linking source code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = ['_coverage_html']