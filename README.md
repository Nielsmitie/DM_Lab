# DM_Lab
The goal of this project is to replicate and extend the results obtained by the [Agnostic Feature Selection](https://www.ecmlpkdd2019.org/downloads/paper/744.pdf) paper.

## Installation
### Install pipenv
```markdown
# for linux adds pipenv to the path
sudo -H pip install -U pipenv

# for Mac
brew install pipenv

# generally use (but then add it to your PATH variable)
pip install --user pipenv
```

### Install dependencies
```shell script
# use --skip-lock because one dependency of tensorflow is currently broken
# installs all packages listed in Pipefile with its dependencies
pipenv install --skip-lock

# installs a new package
pipenv install [packages sepearated by space] --skip-lock
```

### Use pipenv
1. Run the installation command which creates a virtual env file somewhere.
2. Run a script
```shell script
# replace file with pipeline
pipenv run python -m [file]

# keep pipenv shell open
pipenv shell
```

Otherwise use it with PyCharm.
1. Install pipenv && run pipenv install (as seen above)
1. Close PyCharm
2. Delete the .idea folder in the project directory
3. Reopen PyCharm in this Project
4. The pipenv should be detected automatically by PyCharm

Verify this in File -> Settings -> Project: .. -> Project Interpreter
On the top there should a Pipenv project interpreter.

If not click on the gear and add. Choose a virtual environment and then add an existing one. Find the
folder with the pipenv (on linux ~/.local/share/virtualenvs/DM_Lab[hash]/bin/python37) and select the
interpreter than close the window and wait for the scan to happen.
