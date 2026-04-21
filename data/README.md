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

## Important

* Only a few sample episodes are provided here
* The code will not reproduce full results without the complete dataset
* You must provide your own dataset following the same structure

---

## Usage

Make sure:

* Episodes are placed inside `data/`
* The fold file exists in `data/folds/`
* Episode names in the fold file match the directory names
