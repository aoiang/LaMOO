# sphinx API reference

This file describes the sphinx setup for auto-generating the botorch API reference.


## Installation

**Requirements**:
- sphinx >= 2.0
- sphinx_autodoc_typehints

You can install these via `pip install sphinx sphinx_autodoc_typehints`.


## Building

From the `botorch/sphinx` directory, run `make html`.

Generated HTML output can be found in the `botorch/sphinx/build` directory. The main index page is: `botorch/sphinx/build/html/index.html`


## Structure

`source/index.rst` contains the main index. The API reference for each module lives in its own file, e.g. `models.rst` for the `botorch.models` module.
