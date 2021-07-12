
# README

## Environment

We require the pandas library to manipulate the dataset more easily. If you have pandas >= 1.2.5, you don't need to do the following steps to create a virtual environment and download pandas.

```console
foo@bar:~/language_id$ python -m venv venv
foo@bar:~/language_id$ source venv/bin/activate
foo@bar:~/language_id$ pip install -r requirements.txt
```

## Training
To show the help information for `train.py`:
```console
foo@bar:~/language_id$ python train.py --help
usage: train.py [-h] [--k K] [--min_n MIN_N] [--max_n MAX_N] [--res RES]
                [--lang_n LANG_N | --lang_p LANG_P | --top10]

Train an ensemble of three Language Identification models with a hard voting
classifier

optional arguments:
  -h, --help       show this help message and exit
  --k K            Limits the size of the language profiles to only the top-k
                   most frequent ones (default: 300)
  --min_n MIN_N    Minimum number of characters in the n-grams (default: 1)
  --max_n MAX_N    Maximum number of characters in the n-grams (default: 7)
  --res RES        Filename for the language specific and average accuracy
                   (default: Results.csv)
  --lang_n LANG_N  Number of languages to train on, randomly chosen
  --lang_p LANG_P  Percentage of languages to train on, randomly chosen
  --top10          Train on the top 10 spoken languages in Europe
```
The last three arguments, namely *lang_n, lang_p and top10* are mutually exclusive. If none are supplied then the model will train on all languages present in the dataset. *Beware, this might take a while!*

The command below will run the algorithm on the top 10 spoken languages in Europe and will generate three json files and one csv file containing the accuracy results. The json files contain the language profiles of the three different models in the ensemble, which can be loaded again for predicting.
```console
foo@bar:~/language_id$ python train.py --top10
```

If you want more control, you can modify the list in `top10.py` to include languages of interest. The list of possible languages can be found in the file: `data/labels.csv`, the column with the names in English. One can put any language from there in the language list in `top10.py`

## Prediction
After training and generating the json files, we can load those and try predicting our own text. 
```console
foo@bar:~/language_id$ python predict.py --help
usage: predict.py [-h] [--k K] [--min_n MIN_N] [--max_n MAX_N]

Predicts the language of text

optional arguments:
  -h, --help     show this help message and exit
  --k K          Limits the size of the language profiles to only the top-k
                 most frequent ones (default: 300)
  --min_n MIN_N  Minimum number of characters in the n-grams (default: 1)
  --max_n MAX_N  Maximum number of characters in the n-grams (default: 7)
```

Assuming the training was done with the default parameters,  we can try the predicting the language of a text from the console as below:
```console
foo@bar:~/language_id$ echo "Text we want to predict" | python predict.py
English
```
Or if you have a txt document for e.g. `test.txt`:
```console
foo@bar:~/language_id$ python predict.py < test.txt
Language
```
As we can see, we get the prediction of the language.