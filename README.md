> Created for the course `LIN523`

# BERT-eat-dog-world

<img src="https://raw.githubusercontent.com/skogsgren/bert-eat-dog-world/main/bert-not-eat-dog.png" width=33% height=33%>

> 1.) *"Å fan , visste jag inte om , endast läst hennes böcker om bronsåldern
> och järnåldern, oerhört **kulturberikande**"*
> 
> 2.) *"Inte otroligt, besökte det lokala badet igår, 50% icke-svenskar,
> vattenpipor, ﬁck njuta av **kulturberikande** musik... som låter värre än
> Håkan Hellströms värsta låtar"*

Dogwhistle identification (e.g. between the interpretation in 1 and 2), model
using a transformer model (BERT) and active learning implemented by using the
`modAL` library in concordance with the `skorch` API to connect it with 🤗
transformers. The final $f_1$ score I managed to achieve was $0.85$. Dataset
was gathered, and annotation for the original dataset was performed using
[skogsgren/lin354-annotation-ui](https://github.com/skogsgren/lin354-annotation-ui)

[**Live demo on 🤗 spaces**](https://huggingface.co/spaces/skogsgren/LIN532-dogwhistle-identification)

Although the program was created with dogwhistle identifiation in mind, it
should work for active learning in concordance with any text classification
purpose, as long as the conventions to how your data is formatted remain the
same (see data below), and the model is similar in architecture to BERT.
Hyper-parameters can be adjusted accordingly in the `hyper_parameters.json`
(like choosing epochs etc). The model can also be changed, however whether or
not any transformer model capable of text classification would work is not
something I've tested.

The code here is presented as Python scripts, although I recommend running this
either on a computer with a powerful GPU, or (better) as a notebook on e.g.
Kaggle (or at least free). Because active learning requires you to sit there
and annotate data the model is most unsure of, you're gonna spend a lot of time
waiting otherwise.  The code above can be found in notebook form on Kaggle
*[here](https://www.kaggle.com/code/skogsgren/lin523-sparse/notebook)*,
although if you want to use that for your project you're gonna have to edit a
bit more than what you would have to if you've used the python scripts above.

I'll also include *[this notebook](https://www.kaggle.com/code/skogsgren/lin523-experimentation-station/notebook)*
(also on Kaggle), which contains all the different experimentations I performed
in order to get to the conclusions present in the code on this repo, as well as
many of the motivations of *why* certain things were performed but not others.
Certain implementations (e.g. how the dataset is split and imported) in that
notebook look a bit difference than what they do here on this repo (less
sophisticated imo), however they are mostly functionally equivalent.

## Data

Data needs to be provided in tsv files where there must be at least two columns
present: 'text', and 'label'.

Two additional scripts are presents in the scripts folder: one for splitting
into proportional dev/test/train-sets (important for my usecase since the
datasets are very skewed), as well as one which appends manual annotations from
the pool to a specified dataset (important because the dev/test set in my small
initial dataset only had one instance of the more rare interpretation, which
risks making the $f_1$ score less meaningful).

## Instructions

If you just want to use the data provided here then the following commands
will install the necessary requirements, fit the model, and kick you right into
the process of active learning! When you're finished annotating just exit with
<ctrl+c> and the model will be saved automatically in your folder as
`model.pkl`, with the program also then printing metrics for both the test set
and unseen data (if provided). Running the program again with the same
command should import the model instead, allowing you to continue with the
process of active learning.

```bash
pip3 install -r requirements.txt

python3 main.py -train data/dg_train.tsv -dev data/dg_dev.tsv \
-test data/dg_test.tsv -pool data/pool_dg.tsv -unseen data/unseen_dg.tsv
```

## Code structure

The code was created gradually through Jupyter notebooks, and then gathered at
the end to create the python scripts above. The code structure of the notebooks
are standard: 1.) preprocessing, 2.) preparing model, 3.) training model, 4.)
active learning, 5.) evaluation. The python scripts have the following
structure:

```
activebert.py:
    Contains class for an active learner using BERT. The query_loop method
    loops until the batch size is larger than available from the unannotated
    dataset. This means that it most likely will loop forever (unless you're a
    very diligent annotator).
preprocessing.py:
    Contains class for dataloader, i.e. a class that takes tsv files as input
    and structures them in a way that is easily accessible for the active
    learner.
main.py:
    Takes filenames as input from the command line and then creates an active
    learner using those filenames by first creating a dataloader. Performs
    active learning using the query_loop method, and deals with <ctrl+c> to
    exit safely, saving the model to disk as well as simultaneously printing
    metrics.
```

All in all, pretty standard stuff, I reckon.

## Further improvements

This current approach completely disregards contextual factors outside of the
immediate linguistic context when it comes to text classification. If one takes
dogwhistle identification for tweets as an example, then one would ideally want
not to only look at individual tweets, but also at the thread in which it
occurs, as well as the twitter history of that particular individual (if not
also including other ideas from sociolinguistics, like socioeconomic status,
political affliation, etc) This better reflects how at least I would determine
whether or not something is a dogwhistle or not in the wild.
