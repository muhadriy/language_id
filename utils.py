import os
import re
import string
import pandas as pd
from collections import deque


def load_data(data_dir='data', languages=None, dev_test_split=0.5, random_seed=1337):
    """
    Loads the data, splitting the test file into a dev and test set using stratified sampling of the languages.

    Parameters
    ----------
    data_dir : str
        Directory of the data to load
    languages : {None, list, int, float}, optional
        None : Default, load all languages from the data
        list : List of languages e.g ['English', 'German', ...] to load from the data
        int : Randomly sample 'int' number of languages for the train/dev/test sets
        float: Value between 0 and 1, the percentage of languages for the train/dev/test sets
    dev_test_split: float, optional
        Value between 0 and 1, setting the dev and test split ratio.
    random_seed: int
        Random seed to set the random state, making the sampling reproducible.

    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
        train_df, dev_df, test_df
    """

    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'), sep=';', keep_default_na=False)
    label_to_lang = dict(zip(labels['Label'], labels['English']))
    with open(os.path.join(data_dir, 'x_train.txt')) as f:
        x_train = [s.strip() for s in f.readlines()]
    with open(os.path.join(data_dir, 'x_test.txt')) as f:
        x_test = [s.strip() for s in f.readlines()]
    with open(os.path.join(data_dir, 'y_train.txt')) as f:
        y_train = [s.strip() for s in f.readlines()]
    with open(os.path.join(data_dir, 'y_test.txt')) as f:
        y_test = [s.strip() for s in f.readlines()]

    train_df = pd.DataFrame(zip(x_train, y_train), columns=['X', 'y'])
    train_df['y'] = train_df['y'].map(label_to_lang)

    test_df = pd.DataFrame(zip(x_test, y_test), columns=['X', 'y'])
    test_df['y'] = test_df['y'].map(label_to_lang)

    if languages is not None:
        lang_list = languages

        if isinstance(languages, int):
            lang_list = labels['English'].sample(n=languages, random_state=random_seed).to_list()

        if isinstance(languages, float):
            lang_list = labels['English'].sample(frac=languages, random_state=random_seed).to_list()

        train_df = train_df.query('y in @lang_list')
        test_df = test_df.query('y in @lang_list')

    train_df = train_df[train_df['X'].apply(filter_)]
    test_df = test_df[test_df['X'].apply(filter_)]

    dev_df = test_df.groupby('y').sample(frac=dev_test_split, random_state=random_seed)
    test_df = test_df[~test_df.index.isin(dev_df.index)]

    return train_df, dev_df, test_df


def filter_(doc):
    """
    Returns true if doc is not an empty string after removing punctuation, digits and whitespaces

    Parameters
    ----------
    doc : str
        Document to check for emptiness.

    Returns
    -------
    bool
        True if the document is empty after preprocessing
    """
    prep_doc = preprocess(doc)
    prep_doc = re.sub('\s+', '', prep_doc)

    return len(prep_doc) > 0


def preprocess(doc):
    """
    Preprocess 'doc' i.e lowercase text, remove punctuation and digits.

    Parameters
    ----------
    doc : str
        Document to preprocess

    Returns
    -------
    str
        Preprocessed document

    """
    prep_doc = doc.lower()
    prep_doc = re.sub('[' + string.punctuation + ']', '', prep_doc)
    prep_doc = re.sub('[' + string.digits + ']', '', prep_doc)

    return prep_doc


def prep_word(doc):
    """
    Preprocess and split words on whitespaces, for word-based models

    Parameters
    ----------
    doc : str
        Document to process

    Returns
    -------
    list
        List of words in the document, preprocessed
    """
    prep_doc = preprocess(doc)
    prep_doc = re.sub('\s+', ' ', prep_doc)
    words = prep_doc.split()

    return words


def prep_ngram(doc, n):
    """
    Preprocess document, create list of character n-grams

    Parameters
    ----------
    doc : str
        Document to process
    n : int
        Number of characters for the n-grams

    Returns
    -------
    deque :
        List-like sequence of character n-grams
    """
    prep_doc = preprocess(doc)
    words = re.sub('\s+', '_', prep_doc)
    words = "_" + words + "_"
    ngrams = deque()
    for i in range(len(words)-n):
        ngrams.append(words[i:i+n])

    return ngrams


def prep_ngram_range(doc, min_n, max_n):
    """
    Preprocess document, create list of character n-grams ranging from min_n to max_n characters

    Parameters
    ----------
    doc : str
        Document to process
    min_n : int
        Minimum number of characters to include
    max_n : int
        Maximum number of characters to include

    Returns
    -------
    deque
        List-like sequence of character n-grams
    """
    prep_doc = preprocess(doc)
    words = re.sub('\s+', '_', prep_doc)
    words = "_" + words + "_"
    ngrams = deque()
    for k in range(min_n, max_n+1):
        for i in range(len(words)-k):
            ngrams.append(words[i:i+k])

    return ngrams
