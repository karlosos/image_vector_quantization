<h1 align="center">Quantization - team project</h1>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Run Tests](https://github.com/karlosos/image_vector_quantization/workflows/Run%20Tests/badge.svg)
![Lint](https://github.com/karlosos/image_vector_quantization/workflows/Lint/badge.svg)

## Development

1. Create virtual environment with `virtualenv .venv`.
2. Activate venv with `source .venv/bin/activate`.
3. Install packages with `pip install -r requirements.txt`.
4. Activate git hooks with `pre-commit install`.
5. Install `vector_quantization` as a package with `pip install -e .`. This will allow doing imports like `from vector_quantization import .`.
4. Go to `vector_quantization` subfolder.
5. Launch application with `python main.py`.

> **IMPORTANT**: on commiting *black* formatter and *flake8* will check code. To enable this checking run command `pre-commit install`.

### Running tests

Run tests with `pytest` in root

## LaTeX document

1. Install LaTeX [https://www.latex-project.org/get/](https://www.latex-project.org/get/)
2. Edit `docs/template.tex` with your editor, for example [TeXMaker](https://www.xm1math.net/texmaker/)

## Pair programming

Install [Live Share extension for VSCode.](https://marketplace.visualstudio.com/items?itemName=MS-vsliveshare.vsliveshare).
