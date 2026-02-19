# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Res-IRF4'
copyright = '2024, Lucas Vivier, CIRED'
author = 'Lucas Vivier'
version = '4.0'
release = '4.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    'nbsphinx',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []
suppress_warnings = ['myst.mathjax']
numfig = True

bibtex_bibfiles = ['articles.bib']
bibtex_reference_style = 'author_year'

# Mock imports for autodoc (avoids needing full conda env for doc builds)
autodoc_mock_imports = [
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
    'cython', 'numexpr', 'bottleneck', 'psutil', 'requests',
    'jinja2', 'PIL', 'cloudpickle', 'colorama',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/CIRED/Res-IRF4",
    "show_prev_next": False,
    "header_links_before_dropdown": 0,
    "navbar_center": [],
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "Zenodo",
            "url": "https://doi.org/10.5281/zenodo.10405492",
            "icon": "fa-solid fa-database",
        },
    ],
}

html_static_path = ['_static']
html_css_files = ['custom.css']
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}
html_permalinks = False

# Linkcheck tuning to reduce transient CI failures while keeping checks strict.
linkcheck_timeout = 15
linkcheck_retries = 2
linkcheck_workers = 5
