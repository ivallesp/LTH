# Lottery-ticket hypothesis experiments
[![Build Status](https://travis-ci.com/ivallesp/LTH.svg?token=9q5PeCRs1A4Ti99y6RdL&branch=develop)](https://travis-ci.com/ivallesp/lth)
[![Code coverage](https://codecov.io/gh/ivallesp/lth/branch/develop/graph/badge.svg?token=MYZ45LQ2RI)](https://codecov.io/gh/ivallesp/lth)

This repository contains a simple and well-tested lottery-ticket hypothesis implementation primer with Keras, to play with. The intention of this project is to provide a starting point for researching around this topic.

## Getting started
1. Install [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/) in your system following the linked official guides.
2. Open a terminal, clone this repository and `cd` to the cloned folder.
3. Run `pyenv install $(.python-version)` in your terminal for installing the required python.
   version
4. Configure poetry with `poetry config virtualenvs.in-project true`
5. Create the virtual environment with `poetry install`
6. Activate the environment with `source .venv/bin/activate`
7. Try running the unit tests with the following command `nosetests`

If you are interested in running code in Jupyter notebooks with the virtual environment that we have just created, run the following command after activating the venv: `python -m ipykernel install --user --name lth`. This command will install a new kernel in jupyter called `lth` pointing to the virtual environment we have just created.

See an example of the LTH code in the notebooks folder.

## Contribution
Feel free to send issues or pull requests if you want to collaborate.

## License
This repository is licensed under MIT license. More info in the `LICENSE` file.
