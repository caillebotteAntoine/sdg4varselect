# pylint: skip-file
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sdg4varselect"
copyright = "2024, Caillebotte"
author = "Caillebotte"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

modindex_common_prefix = ["sdg4varselect."]

autodoc_default_flags = ["members", "inherited-members"]
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]


from importlib.metadata import version

release = version(project)
version = release


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "body_max_width": "none",
    "page_width": "auto",
}


import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
