# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from gmdh.version import __version__ # pylint: disable=wrong-import-position

project = 'gmdh'
copyright = '2022, bauman-team'
author = 'bauman-team'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.doctest', 'sphinx.ext.autodoc', 'numpydoc', 'enum_tools.autoenum', 'sphinx_autodoc_typehints']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

numpydoc_show_class_members = False 
