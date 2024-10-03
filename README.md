# Estimating and Differentiating Programming Skills from Unstructured Coding Exercises via Matrix Factorization - Koli Calling 2024

*Note: This is a stripped down version of our code prepared for publication along with the paper. It provides the implementation but not the actual data used in our experiments*

## Requirements
- python 3.11
- `pipenv` for python virtual env
- setup to run Jupyter Notebooks (e.g. VS-Code + Jupyter Plugins)
- dataset resides in `./data` folder

## Install packages
- `$ pipenv install --dev`

## Files and Folders
- `src/minimal.ipynb`: minimal example notebook to load data and train a model
- `src/algorithms`: implementation of matrix factorization models
- `src/shared`: code for data loading and preparation
- `src/langusage.jar`: JAR of Java project developed by our second author, which is used to parse Java code and retrieve language constructs
