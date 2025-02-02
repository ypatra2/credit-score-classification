# credit-score-classification

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This repository contains a machine learning pipeline for credit score classification using the Kedro framework. It includes data preprocessing, model training with `RandomForestClassifier`, and evaluation steps. The project adheres to best practices in software development and data science for modularity and maintainability.

## Rules and Guidelines

To ensure the best practices are followed:

* Do not remove any lines from the `.gitignore` file provided.
* Ensure your results are reproducible by following the [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention).
* Do not commit data to your repository.
* Do not commit any credentials or local configuration to your repository. Keep all credentials and local configuration in `conf/local/`.

## How to Install Dependencies

Declare any dependencies in `requirements.txt` for `pip` installation. To install them, run:

```
pip install -r requirements.txt
```

## How to Run Your Kedro Pipeline

You can run your Kedro project with:

```
kedro run
```

## How to Test Your Kedro Project

Refer to the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on writing tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, refer to the `.coveragerc` file.

## Project Dependencies

To see and update the dependency requirements for your project, use `requirements.txt`. Install the project requirements with:

```
pip install -r requirements.txt
```

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to Work with Kedro and Notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines`, and `session`.

Jupyter, JupyterLab, and IPython are included in the project requirements by default. After running `pip install -r requirements.txt`, you can use them without additional steps.

### Jupyter

To use Jupyter notebooks in your Kedro project, install Jupyter:

```
pip install jupyter
```

Start a local notebook server with:

```
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, install it with:

```
pip install jupyterlab
```

Start JupyterLab with:

```
kedro jupyter lab
```

### IPython

To run an IPython session:

```
kedro ipython
```

### How to Ignore Notebook Output Cells in `git`

To automatically strip out all output cell contents before committing to `git`, use tools like [`nbstripout`](https://github.com/kynan/nbstripout). Add a hook in `.git/config` with:

```
nbstripout --install
```

> *Note:* Your output cells will be retained locally.

## Package Your Kedro Project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
