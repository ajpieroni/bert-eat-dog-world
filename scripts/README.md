This folder contains two scripts useful for creating and manipulating tsv files
as to fit the convention of having a test/train/dev-set. 

- `test_train_dev_tsv_split.py` takes a tsv file as input and then exports that
  tsv file to three seperate tsv filesaccording to proportion in an 80/10/10
  split.
- `add_manual_annotations_to_tsv.py` takes a json file, a tsv file (which will
  be modified) containing either dev/test or training set, as well as a tsv
  file containing unannotated data (which will also be modified) as input.
  The JSON file contains manually annotated data from the tsv file containing
  unannotated data. The function
  does two things: it removes those datapoints from the tsv file containing
  unannotated data, and it adds it (with the manual annotation from the JSON
  file) to the provided train/test/dev tsv file. This is needed when the
  dataset is very skewed as to give an accurate $f_1$ score.
