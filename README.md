# DM_Lab
The goal of this project is to replicate and extend the results obtained by the [Agnostic Feature Selection](https://www.ecmlpkdd2019.org/downloads/paper/744.pdf) paper.

## Installation
1. Install [Anaconda](https://www.anaconda.com/distribution/)
2. Install prerequisites for [Tensorflow GPU](https://www.tensorflow.org/install/gpu) support
3. Open command prompt
4. `conda env create -f environment.yml`
5. `conda activate dml`
6. `cd scikit-feature`
7. `pip install .`

## Start Jupyter Notebook
1. Open command prompt
2. `conda activate dml`
3. `jupyter notebook`
4. Open "Data Exploration"

## Installation with Pipenv

### Install pipenv


```shell script
# for linux adds pipenv to the path
sudo -H pip install -U pipenv

# installation for Mac
brew install pipenv

# generally use
pip install --user pipenv
# but then you have to add it to your PATH variable
```

Otherwise use magic Google.

### Install dependencies

```shell script
# use --skip-lock because one dependency of tensorflow is currently broken
pipenv install --skip-lock
# installs all packages listed in Pipefile with its dependencies
# install a new package
pipenv install [packages sepearated by space] --skip-lock
```

### Using pipenv

1. Run the installation command which creates a virtual env file somewhere.
2. Run a script
```shell script
pipenv run python -m [file.py]
```

Otherwise use it with PyCharm.
1. Install pipenv && run pipenv install (as seen above)
1. Close PyCharm
2. Delete the .idea folder in the project directory
3. Reopen PyCharm in this Project
4. The pipenv should be detected automatically by PyCharm

Verify this in FIle -> Settings -> Project: .. -> Project Interpreter
On the top there should a Pipenv project interpreter.

If not click on the gear and add. Choose a virtual environment and then add an existing one. Finde the
folder with the pipenv (on linux ~/.local/share/virtualenvs/DM_Lab[hash]/bin/python37) and select the
interpreter than close the window and wait for the scan to happen.
