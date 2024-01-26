# Data 

We have prepared two training datasets based on different physicochemical properties. Both were created from a subset of the ZINC dataset.
These datasets match the dataset utilised in our paper, Generating Property-Matched Decoy Molecules using Deep Learning.

We have also provided several scripts to allow you to use your own dataset.

# To use a provided dataset

To process the provided datasets, run `prepare_data.py`. This allows you to train and validate a model, as well as generate molecules using `DeepCoy.py`.

```
python prepare_data.py
```

# To use your own dataset


## Training

If you want to process your own training dataset, run `prepare_data.py` with the following arguments:

```
python prepare_data.py --data_path PATH_TO_DATA --dataset_name NAME_OF_DATASET --save_dir OUTPUT_LOCATION --reverse
```

The format taken by `prepare_data.py` for training is: 

```
molecule_1 (SMILES) molecule_2 (SMILES)
```

For example:

```
CC1CCC([NH+](C)CC[C@H]2CCC[C@]2([NH3+])CO)CC1 CC[C@@H]1CC[C@@](C[NH3+])(C2(O)CC[NH+](C(C)C)CC2)C1
```

Anything after the first two entries on a line will be ignored.

## Testing/Generating

If you want to prepare your own test set, run `prepare_data.py` with the following arguments:

```
python prepare_data.py --data_path PATH_TO_DATA --dataset_name NAME_OF_DATASET --save_dir OUTPUT_LOCATION
```

The format taken by `prepare_data.py` for generating decoys is:

```
molecule_1 (SMILES)
```

There should be no other entries on a line other than the SMILES string of the molecule to generate decoys for.

You can also prepare multiple files for testing at once using `prepare_dataset.py`:
```
python prepare_dataset.py --data_path PATH_TO_DATA --save_dir OUTPUT_LOCATION
```

Note: To prepare phosphorus containing compounds, you will need to change line 29 of `prepare_data.py` or line 27 of `prepare_dataset.py` as indicated in the respective file.

# Contact (Questions/Bugs/Requests)

Please submit a Github issue or contact either Fergus Imrie or the Oxford Protein Informatics Group (OPIG) [deane@stats.ox.ac.uk](mailto:deane@stats.ox.ac.uk).
