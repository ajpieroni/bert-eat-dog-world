This folder contains two scripts useful for creating and manipulating tsv files
as to fit the convention of having a test/train/dev-set. 

- `test_train_dev_tsv_split.py` takes a tsv file as input and then exports that
  tsv file to three seperate tsv files according to proportion of distribution
  in original tsv file in an 80/10/10 split.
- `add_manual_annotations_to_tsv.py` adds manual annotations (provided in a
  json file) from the large unannotated pool (tsv file) to another tsv file,
  removing it from the larger unannotated pool as it does so. This is performed
  because in certain skewed datasets certain labels have very few occurances,
  which can give overly optimistic $f_1$ scores.
