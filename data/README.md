# Sample Data

This folder contains **a small subset of example episodes** to illustrate the expected data format.

The full dataset is described in the main repository README and will be publicly released upon publication.

---

## Structure

Each episode follows this structure:

```
episode_X/
├── frame_0000
├── frame_0001
├── ...
└── labels_updated.pkl
```

Note: Sample episodes are provided as .zip files and must be extracted before use.

---

## Test Split File

The code requires a fold definition file:

```
data/folds/test_fold.txt
```

This file specifies which episodes are used for evaluation.

### Format

One episode name per line:

```
episode_1
episode_3
```

Episode names must match the folder names exactly.

---

## Configuration

To run the code with the provided sample data, you must correctly set the dataset-related parameters in the configuration file.

The relevant fields are:

```python
# =========================================================
# DATASET
# =========================================================
DATASET_NAME = 'Dataset'

FIVE_FOLD = False
FOLDS = ["example_fold.txt"]

ANNOTATIONS_FOLDER = "data/example/"
LABELS_FILE_NAME = "labels_updated.pkl"
```

### Notes

* `FOLDS` specifies the file containing the test split.
  This should match a file inside `data/folds/` (e.g., `example_fold.txt`).

* `ANNOTATIONS_FOLDER` is the path to the dataset root directory containing the episode folders.
  For the provided samples, this should point to `data/example/`.

* `LABELS_FILE_NAME` must match the annotation file inside each episode folder (`labels_updated.pkl`).

* `FIVE_FOLD` should be set to `False` when using a single split (as in the provided examples).

---

## Important

* Only a few sample episodes are provided here
* The code will not reproduce full results without the complete dataset
* You must provide your own dataset following the same structure
* Episode names in the fold file must exactly match the directory names

---

## Usage

Before running the code, ensure that:

* Episodes are placed inside the directory specified by `ANNOTATIONS_FOLDER`
* The fold file exists in `data/folds/`
* Configuration paths match your local setup
