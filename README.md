# An Annotated Dataset of 1,000 Candidate Errors in Premodern Greek

This repository contains the data and code used for the paper "An Annotated Dataset of 1,000 Candidate Errors in Premodern Greek and Baselines for Detecting Them." The dataset and baselines provided here are designed to accelerate the discovery of real errors in premodern texts using machine learning methods.

## Dataset Files

The `dataset_files` directory contains two annotated files: `errors1.txt` and `errors5.txt`. We randomly split the works of Michael Psellos into five parts, and the annotations provided here correspond to parts 1 and 5, with the remaining parts currently in progress. Each of these files contains 500 line-separated dictionaries.

To load the data from these files, you can do the following:

```python
import ast

filepath = 'path/to/errors1.txt'  # or 'path/to/errors5.txt'

with open(filepath, 'r') as file:
    reports = [ast.literal_eval(line.strip()) for line in file]

## Dictionary Structure

Each dictionary in the dataset contains the following keys:

- **Transmitted Word**: The word as it appears in the transmitted text.
- **Word Index in Text**: The index of the word in the array text.split().
- **Model-Suggested Alternative**: The alternative word suggested by the model.
- **Label**: The domain expert's label indicating whether the flag is a "GOOD FLAG." (a genuine error) or a false positive.
- **Notes**: Additional notes from the domain expert, providing context or further details about the flag.
- **Text**: A snippet of the surrounding text where the transmitted word is found.



