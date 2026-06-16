[한국어](README.md) | English

# Improving EEG Drowsiness-Detection Classification with Deep Learning

This is the EEG-based drowsiness detection research I did as an undergraduate researcher at the ICaN Lab, Department of Software, Dongseo University. The goal was to classify drowsiness states from EEG (brainwave) signals and push that classification performance higher with deep learning.

What lives in this repository is the early preprocessing stage of the project, specifically the work of reconstructing raw recordings into a shape that a model can train on. The classification model itself is not yet committed as code; what you see here is the groundwork that comes before it.

## Data

The source data are EEG recordings in MATLAB `.mat` format, loaded and handled with `scipy.io.loadmat`. The actual data (`/data`) is excluded from the repository because of its size.

The data is split into two labels.

- Two labels, `sb` and `sg`, with 11 files each (`sb1`~`sb11`, `sg1`~`sg11`)
- Per person, the `sb` and `sg` recordings are paired together

The signal itself is structured as follows.

- Of the 68 total channels, 1–64 are EEG, 65–66 are EOG, and 67–68 are ECG. This study uses only the 64 EEG channels.
- The sampling rate is 512 Hz.
- The point where the event value equals `1` marks the start of measurement; everything before it is dropped and only the signal from that start point is kept.
- A single recording runs roughly 30 minutes, which comes to about 921,600 samples from the start point.

## Data reconstruction

The core task is reconstructing one long 1-D time series into a 3-D array that is easier for a model to work with.

The signal is sliced into one-minute trials (`SR × 60` = 30,720 samples) and stacked into a `[trials × channels × time]` shape. For a 30-minute recording this becomes `[30 × 64 × 30720]`.

```
[trials × channels × time-series] = [30 × 64 × 30720]
```

The `EEGData` class takes a single file and handles loading, start-point alignment, and one-minute slicing in one pass. Reconstruction runs internally when the object is created, so the resulting `ProcessedData` is ready to use right away.

From there, per person, the `sb` and `sg` trials are concatenated to form a `[trials*2 × 64 × 30720]` array, stored in `person_dict` as `p1`~`p11`. When the two labels have a different number of trials, the larger one is trimmed down to match the smaller.

To confirm the reconstruction is correct, the result is compared value-for-value against a separately MATLAB-generated `reconstruction.mat` using `np.array_equal`.

## Repository layout

```
src/
  EEG-Research-Main.ipynb              # Main notebook: EEGData class, per-person reconstruction
  Data-Reconstrion-Single-Data.ipynb  # Single-file reconstruction worked through step by step
  TestCodes.ipynb                      # Experiments and test code for the reconstruction logic
img/
  matlab_data_view.png                 # The raw data structure as seen in MATLAB
```

`Data-Reconstrion-Single-Data.ipynb` walks through a single file one step at a time, from type checking to event splitting to slicing. The main notebook is what you get when that process is wrapped into a class and applied across multiple files.

## Stack

- Python (Jupyter Notebook)
- NumPy
- SciPy (`scipy.io.loadmat`)

## Notes

The code follows PEP8 and Google-style docstrings as a base, alongside the personal conventions noted in the first notebook cell (docstring tags such as `@`, `&`, `$`, `~`, `<T>`). This was written at the undergraduate research stage and there is plenty left to clean up, but I keep it here as a record of hands-on work with EEG signals and of designing a preprocessing pipeline myself.
