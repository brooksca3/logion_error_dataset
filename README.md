# An Annotated Dataset of Errors in Premodern Greek and Baselines for Detecting Them

This repository contains the data and code used for the paper "An Annotated Dataset of Errors in Premodern Greek and Baselines for Detecting Them". The dataset and baselines provided here are designed to accelerate the discovery of real errors in premodern texts using machine learning methods.

## Dataset Files

The `dataset_files` directory contains two annotated files: `errors_split_1.json` and `errors_split_5.json`. We randomly split the works of Michael Psellos into five parts, and the annotations provided here correspond to parts 1 and 5, with the remaining parts currently in progress. Each of these files contains 500 line-separated dictionaries.

To load the data from these files, you can do the following:

```python
import json

filepath = 'path/to/errors_split_1.txt'  # or 'path/to/errors_split_5.txt'

with open('path/to/errors_split_1.json', 'r') as file:
    combined_reports1 = json.load(file)
```
We additionally include the file `random_assumed_true_negatives.json` which contains the words randomly selected from the non-flags and are presumed to be non-erroneous. We include these examples in our evaluation to mitigate the distribution shift incurred by our approach for oversampling true errors. 

## Dictionary Structure

Each dictionary in the dataset contains the following keys:

- **Transmitted Word**: The word as it appears in the transmitted text.
- **Word Index in Text**: The index of the word in the array `text.split()`.
- **Model-Suggested Alternative**: The alternative word suggested by the model.
- **Label**: The domain expert's label indicating the nature of the candidate error.
- **Notes**: Additional notes from the domain expert, providing context or further details about the flag.
- **Text**: A snippet of the surrounding text where the transmitted word is found.

### Labels

The **Label** key can have one of the following values:

- **GOOD FLAG.** This indicates that the domain expert has identified a genuine error in the transmitted text.
- **BAD.** This indicates that the flagged word is not a genuine error.
- **PLAUSIBLE FLAG.** This indicates that the flag seems legitimate, but further work is needed to be sure.
- **UNCERTAIN.** This indicates that further work is needed to determine if the flag is a genuine error.
- **BAD DATA.** This indicates that the error resulted from issues in the authors' data assembling, cleaning, or standardization.
- **EDITORIAL.** This indicates that the flagged issue is not a problem with the text, but a situation where different editorial decisions (such as punctuation or spacing) can be valid.



