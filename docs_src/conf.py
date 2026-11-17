# Sphinx configuration for the DMDO documentation.
#
# Build (from the repository root):
#   pip install -r docs_src/requirements.txt
#   sphinx-build -b html docs_src docs
#
# The rendered site is written straight into ``docs/`` because GitHub Pages
# serves the documentation from that folder.

import os
import sys
from datetime import date

try:
  import tomllib
except ModuleNotFoundError:  # Python < 3.11
  tomllib = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# -- Project information -----------------------------------------------------

project = "DMDO"
author = "Ahmed H. Bayoumy"
copyright = f"2022-{date.today().year}, {author}"

release = "2612.1"
if tomllib is not None:
  with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
    release = tomllib.load(f)["project"]["version"]
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_proof",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.pseudocode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts"]
numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s",
                 "code-block": "Listing %s"}

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"show-inheritance": False}
# Heavy optional dependencies are not needed to read the API: DMDO imports them lazily or only for features that the
# documentation does not cover, so the docs build without them.
autodoc_mock_imports = ["torch", "OMADS", "samplersLib", "dash", "plotly", "paramiko", "dash_bootstrap_components"]
napoleon_use_rtype = False

# pseudocode.js (the ``pcode`` algorithm blocks) waits for MathJax 3's startup
# promise; with MathJax 4 the algorithms are never rendered.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"DMDO {release}"
html_logo = "_static/DMDO_logo.png"
html_favicon = None
html_static_path = ["_static"]
# Redirects from the page names of the earlier documentation.
html_extra_path = ["_extra"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

html_theme_options = {
    "github_url": "https://github.com/Ahmed-Bayoumy/DMDO",
    "logo": {"image_dark": "_static/DMDO_logo.png"},
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    # The "On this page" contents sit in the left sidebar, below the navigation.
    "secondary_sidebar_items": [],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_sidebars = {"**": ["sidebar-nav-bs", "page-toc"]}
html_context = {"default_mode": "light"}
