# Data Folder

This folder contains **sample data** to illustrate the expected format of the dataset.

The full dataset will be publicly released upon publication.

## Structure

* `sample_*/`: Example data instances
* `folds/test_fold.txt`: Specifies which samples are used for testing

## Fold File Format

The file `test_fold.txt` contains a list of sample identifiers (one per line) corresponding to the test split.

Example:
sample_1
sample_3

## Important

The code expects:

* A valid fold file in `data/folds/`
* Data samples matching the identifiers listed in the fold file

If you use your own dataset, make sure to follow the same structure.
