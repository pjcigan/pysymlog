# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


### Mock import modules
#import sys
#from unittest.mock import MagicMock

#class Mock(MagicMock):
#    @classmethod
#    def __getattr__(cls, name):
#        return MagicMock()
#
#MOCK_MODULES = ['setuptools', 'multicolorfits', 'pyqt5', 'PyQt5','PyQt5.QtGui','PyQt5.QtWidgets', 'pyface.qt','pyface.qt.QtCore', 'traitsui', 'traits', 'setup', 'astroquery', 'astroquery.skyview', 'astroquery.sdss', 'astroquery.simbad', 'astropy.coordinates', 'astropy.coordinates.spectral_coordinate',]
#sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysymlog'
copyright = '2024, Phil Cigan'
author = 'Phil Cigan'

# The short X.Y version
version = '1.0'
# The full version, including alpha/beta/rc tags
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    #'autoapi.extension',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    #'recommonmark', #now deprecated.  Use myst_parser instead.
    'myst_parser',
    'sphinx.ext.napoleon',
    'nbsphinx', #For using jupyter notebooks
    #'myst_nb', #Alternative for using jupyter notebooks
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']
#nbsphinx_custom_formats = { ".md": ["jupytext.reads", {"fmt": "mystnb"}], }

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme='sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
         'custom.css',
         ]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
#htmlhelp_basename = 'pysymlogdoc'


