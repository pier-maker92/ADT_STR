# Dataset Augmentation Pipeline

End-to-end pipeline for generating, organizing, and converting the augmented one-shot dataset.

## 1. CLAP Scores Generation & Binning
Computes the cosine similarity between the reference dataset and the sample pack files. Sorts and organizes the samples into percentage bins within a new `{reference_root}_clap_augmented` directory.

```bash
python data_modules/augment_data_with_CLAP.py configs/preprocess/clap.yaml --num_bins 10
```

## 2. Original Dataset Consolidation
Copies the original reference dataset samples into the augmented dataset directory to maintain a single source of truth.

```bash
python data_modules/copy_originals_to_augmented.py --source <original_reference_root> --dest <augmented_root>
```

## 3. Fast I/O Conversion (HDF5)
Converts the entire audio dataset into a single compressed binary file (`.hdf5`) to optimize I/O performance during training.

```bash
python data_modules/convert_augmented_to_hdf5.py <augmented_root> <output_file_prefix> --sample_rate 44100 --overwrite
```
