import numpy as np
from collections import Counter
from functools import partial
import json
from utils import prep_word, prep_ngram, prep_ngram_range
from abc import ABC, abstractmethod
import os


class LanguageProfiles:
    """Language profiles and scoring functions"""

    def __init__(self, score_fn='Default'):
        """
        Initialize the profiles and set the scoring function

        Parameters
        ----------
        profiles : dict
            dict mapping Language -> Counter

        score_fn : function
            Function to score similarity between a document and language profile
        """

        self.profiles = None

        self.score = self.count
        self.maximize = True

        if score_fn == 'avg_overlap':
            self.score = self.average_overlap
            self.maximize = True

        if score_fn == 'rbo':
            self.score = self.rbo_min
            self.maximize = True

        if score_fn == 'rank':
            self.score = self.rank_score
            self.maximize = False

    def set_profiles(self, profiles):
        """
        Set the language profiles to profiles
        Parameters
        ----------
        profiles : dict
            dict Language -> {word/ngram -> count}

        Returns
        -------
            dict Language -> Counter({word/ngram -> count})
        """
        self.profiles = profiles

        if self.profiles is not None:
            for key, val in self.profiles.items():
                self.profiles[key] = Counter(self.profiles[key])

    def count(self, doc_counts, lang):
        """Counts the words/ngrams in common"""
        return sum((doc_counts & lang).values())

    def average_overlap(self, doc_counts, lang):
        """
        Calculates the average overlap between a language profile and the document i.e.:
        averages the size of intersection with growing rank

        Parameters
        ----------
        doc_counts : Counter
            dict-like Counter, with words/ngrams of the document as keys and counts as values

        lang : Counter
            dict-like Counter, with words/ngrams of the language as keys and counts as values

        Returns
        -------
        float
            Average Overlap score
        """
        d = min(len(doc_counts), len(lang))
        S_d = np.array([x[0] for x in doc_counts.most_common(d)])
        L_d = np.array([x[0] for x in lang.most_common(d)])

        rank_overlap = np.array([len(np.intersect1d(S_d[:i], L_d[:i], assume_unique=True)) for i in range(d)])
        avg_overlap = np.mean(rank_overlap)

        return avg_overlap

    def rank_score(self, doc, lang):
        """
        Out-of-order score
        See Also: Cavnar and Trenkle (1994) https://www.let.rug.nl/vannoord/TextCat/textcat.pdf

        Parameters
        ----------
        doc : dict
            dict with n-grams of the document as a key and the ranking as value
        lang : dict
            dict with n-grams of the language as a key and the ranking as value

        Returns
        -------
        float
            Out-of-order score

        """
        penalty = len(doc)+1
        score = [abs(lang[x]-doc[x]) if x in doc else penalty for x in lang.keys()]

        return sum(score)

    def rbo_min(self, doc_counts, lang, p=0.99):
        """
        Rank biased overlap: A similarity measure for indefinite rankings by Webber & Moffat 2010
        See Also: http://codalism.com/research/papers/wmz10_tois.pdf

        Parameters
        ----------
        doc_counts : Counter
            dict-like Counter, with words/ngrams of the document as keys and counts as values

        lang : Counter
            dict-like Counter, with words/ngrams of the language as keys and counts as values
        p : float
            Determines the contribution of the top-d weights to the score, 0 < p < 1

        Returns
        -------
        float
            Lower bound on the rank based overlap score
        """
        d = min(len(doc_counts), len(lang))
        S_d = np.array([x[0] for x in doc_counts.most_common(d)])
        L_d = np.array([x[0] for x in lang.most_common(d)])

        X_d = len(np.intersect1d(S_d, L_d, assume_unique=True))

        d_s = np.arange(1, d+1)
        weights = (p**d_s)/d_s

        X_s = np.array([len(np.intersect1d(S_d[:i], L_d[:i], assume_unique=True)) for i in d_s]) -X_d
        rbo_score = (1-p)/p * (np.sum(X_s*weights) - X_d*np.log(1-p))

        return rbo_score

    def predict(self, doc, preprocess, f=None):
        """
        Predict the language of doc by comparing it with every language profile and choosing the highest/lowest scoring
        language depending on the scoring function

        Parameters
        ----------
        doc : str
            Document to predict the language of
        preprocess : function
            Function to preprocess the document
        f : function
            Function to process the scores if needed

        Returns
        -------
        str
            Predicted language of the document
        """
        scores = {}
        doc_score = Counter(preprocess(doc))
        if f is not None:
            doc_score = f(doc_score)

        for lang in self.profiles.keys():
            scores[lang] = self.score(doc_score, self.profiles[lang])

        return sorted(scores.items(), key=lambda x: x[1], reverse=self.maximize)[0][0]


class LanguageIdentifier(ABC):
    """Abstract base class for Language Identifier implementations"""

    def __init__(self, score_fn):
        """Initialize LanguageProfiles with the provided scoring function"""
        self.lang_prof = LanguageProfiles(score_fn=score_fn)

    @abstractmethod
    def generate_lp(self, df):
        """Generate the profile for one language"""
        pass

    def train(self, df):
        """
        Checks the format of the df and generates the language profiles of each language

        Parameters
        ----------
        df : DataFrame
            Contains the documents with their language labels
        """
        if 'X' not in df:
            raise ValueError('Documents (str) not in the df.')

        if 'y' not in df:
            raise ValueError('Language labels not in the df.')

        self.lang_prof.profiles = dict(df.groupby('y').apply(lambda x: self.generate_lp(df=x)))

    @abstractmethod
    def predict(self, df):
        """Predict the language of the documents in df"""
        if self.lang_prof.profiles is None:
            raise ValueError('Language Profiles have not been generated. Call train(df) first!')

    @abstractmethod
    def predict_doc(self, doc):
        """Predict the language of a single document"""

    def accuracy(self, df):
        """Calculates the average accuracy between the true language and predicted language"""
        if 'y' not in df:
            raise ValueError('Language labels not in the df.')

        if 'y_pred' not in df:
            raise ValueError('Predicted language labels not in the df. Call predict(df) first!')

        return (df['y'] == df['y_pred']).mean()

    def lang_accuracy(self, df):
        """
        Calculates the language specific accuracy

        Parameters
        ----------
        df : DataFrame

        Returns
        -------
        dict
            dict: Language -> Accuracy
        """
        if 'y' not in df:
            raise ValueError('True labels not in the df.')

        if 'y_pred' not in df:
            raise ValueError('Predicted labels not in the df. Call predict(df) first!')

        return dict(df.groupby('y').apply(lambda group: (group['y'] == group['y_pred']).mean()))

    def to_json(self, filename='lp.json', save_dir='profiles'):
        """
        Store the language profiles to json

        Parameters
        ----------
        filename : str
            Filename ending with .json

        Returns
        -------

        """
        json.dump(self.lang_prof.profiles, open(os.path.join(save_dir,filename), 'w'))
        print("Language Profiles saved as", os.path.join(save_dir,filename))

    def load(self, filename='lp.json', save_dir='profiles'):
        """
        Load the stored language profiles from json
        Parameters
        ----------
        filename : str
            Filename to load from, ending with .json

        Returns
        -------

        """
        with open(os.path.join(save_dir,filename)) as jf:
            profiles = json.load(jf)
        self.lang_prof.set_profiles(profiles)


class WordLI(LanguageIdentifier):
    """Word-based Language Identification"""

    def __init__(self, k=50, score_fn='Default'):
        """
        Initialize

        Parameters
        ----------
        k : int
            Number of top frequent words to keep in each profile, 0 to keep everything
        score_fn : function
            Function to score similarity between a document and language profile
        """
        super().__init__(score_fn=score_fn)
        self.k = k

    def generate_lp(self, df):
        """
        Generate the language profile of a language

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the documents of a language

        Returns
        -------
        Counter
            dict-like Counter, with words as keys and counts as values
        """
        documents = df['X'].to_numpy()
        counts = Counter()
        for doc in documents:
            counts.update(prep_word(doc))

        if self.k > 0:
            counts = Counter(dict(counts.most_common(self.k)))
        return counts

    def predict(self, df):
        """
        Predict the language for each document in df, storing the prediction in the 'y_pred' column

        Parameters
        ----------
        df : DataFrame
            Documents to predict

        Returns
        -------

        """
        super().predict(df)

        df['y_pred'] = df['X'].map(lambda x: self.lang_prof.predict(x, prep_word))

    def predict_doc(self, doc):
        """
        Predict the language of one document

        Parameters
        ----------
        doc : str
            Document

        Returns
        -------
        str
            Predicted language
        """
        return self.lang_prof.predict(doc, prep_word)


class NGramLI(LanguageIdentifier):
    """Character n-gram based Language Identifier"""

    def __init__(self, k=50, n=4, score_fn='Default'):
        """
        Initialize

        Parameters
        ----------
        k : int
            Number of top frequent n-grams to keep in each profile, 0 to keep everything
        n : int
            Number of characters for the n-grams
        score_fn : function
            Function to score similarity between a document and language profile
        """
        super().__init__(score_fn=score_fn)
        self.k = k
        self.n = n

    def generate_lp(self, df):
        """
        Generate the language profile of a language

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the documents of a language

        Returns
        -------
        Counter
            dict-like Counter, with n-grams as keys and counts as values
        """
        documents = df['X'].to_numpy()
        counts = Counter()
        for doc in documents:
            counts.update(prep_ngram(doc, self.n))

        if self.k > 0:
            counts = Counter(dict(counts.most_common(self.k)))
        return counts

    def predict(self, df):
        """
        Predict the language for each document in df, storing the prediction in the 'y_pred' column

        Parameters
        ----------
        df : DataFrame
            Documents to predict

        Returns
        -------

        """
        super().predict(df)

        preprocess = partial(prep_ngram, n=self.n)
        df['y_pred'] = df['X'].map(lambda x: self.lang_prof.predict(x, preprocess))

    def predict_doc(self, doc):
        """
        Predict the language of one document

        Parameters
        ----------
        doc : str
            Document

        Returns
        -------
        str
            Predicted language
        """
        preprocess = partial(prep_ngram, n=self.n)
        return self.lang_prof.predict(doc, preprocess)


class NGramRangeLI(LanguageIdentifier):
    """Character n-gram based Language Identifier, with characters from min_n to max_n"""

    def __init__(self, k=50, min_n=1, max_n=5, score_fn='Default'):
        """
        Initialize

        Parameters
        ----------
        k : int
            Number of top frequent n-grams to keep in each profile, 0 to keep everything
        min_n : int
            Minimum number of characters in the n-grams
        max_n : int
            Maximum number of characters in the n-grams
        score_fn : function
            Function to score similarity between a document and language profile
        """
        super().__init__(score_fn=score_fn)
        self.k = k
        self.min_n = min_n
        self.max_n = max_n

    def generate_lp(self, df):
        """
        Generate the language profile of a language

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the documents of a language

        Returns
        -------
        Counter
            dict-like Counter, with n-grams as keys and counts as values
        """
        documents = df['X'].to_numpy()
        counts = Counter()
        for doc in documents:
            counts.update(prep_ngram_range(doc, min_n=self.min_n, max_n=self.max_n))

        if self.k > 0:
            counts = Counter(dict(counts.most_common(self.k)))
        return counts

    def predict(self, df):
        """
        Predict the language for each document in df, storing the prediction in the 'y_pred' column

        Parameters
        ----------
        df : DataFrame
            Documents to predict

        Returns
        -------

        """
        super().predict(df)

        preprocess = partial(prep_ngram_range, min_n=self.min_n, max_n=self.max_n)
        df['y_pred'] = df['X'].map(lambda x: self.lang_prof.predict(x, preprocess))

    def predict_doc(self, doc):
        """
        Predict the language of one document

        Parameters
        ----------
        doc : str
            Document

        Returns
        -------
        str
            Predicted language
        """
        preprocess = partial(prep_ngram_range, min_n=self.min_n, max_n=self.max_n)
        return self.lang_prof.predict(doc, preprocess)


class RankNGramRangeLI(LanguageIdentifier):
    """
    Out-of-order ranked character n-gram Language Identifier, with characters from min_n to max_n
    See Also: Cavnar and Trenkle (1994) https://www.let.rug.nl/vannoord/TextCat/textcat.pdf
    """

    def __init__(self, k=300, min_n=1, max_n=5):
        """
        Initialize

        Parameters
        ----------
        k : int
            Number of top frequent n-grams to keep in each profile, 0 to keep everything
        min_n : int
            Minimum number of characters in the n-grams
        max_n : int
            Maximum number of characters in the n-grams
        score_fn : function
            Function to score similarity between a document and language profile
        """
        super().__init__(score_fn='rank')
        self.k = k
        self.min_n = min_n
        self.max_n = max_n

    def rank_counts(self, counts):
        """
        Ranks the n-grams based on their counts, with the most frequent being the highest ranked

        Parameters
        ----------
        counts : Counter
            dict-like Counter, with n-grams as keys and counts as values

        Returns
        -------
        dict
            dict with n-grams as keys and rankings as values

        """
        ranking = {}

        for i, x in enumerate(counts.most_common(len(counts))):
            ranking[x[0]] = i + 1

        return ranking

    def generate_lp(self, df):
        """
        Generate the language profile of a language

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the documents of a language

        Returns
        -------
        dict
            dict with n-grams as keys and rankings as values
        """
        documents = df['X'].to_numpy()
        counts = Counter()
        for doc in documents:
            counts.update(prep_ngram_range(doc, min_n=self.min_n, max_n=self.max_n))

        if self.k > 0:
            counts = Counter(dict(counts.most_common(self.k)))

        ranking = self.rank_counts(counts)
        return ranking

    def predict(self, df):
        """
        Predict the language for each document in df, storing the prediction in the 'y_pred' column

        Parameters
        ----------
        df : DataFrame
            Documents to predict

        Returns
        -------

        """
        super().predict(df)

        preprocess = partial(prep_ngram_range, min_n=self.min_n, max_n=self.max_n)
        df['y_pred'] = df['X'].map(lambda x: self.lang_prof.predict(x, preprocess, self.rank_counts))

    def predict_doc(self, doc):
        """
        Predict the language of one document

        Parameters
        ----------
        doc : str
            Document

        Returns
        -------
        str
            Predicted language
        """
        preprocess = partial(prep_ngram_range, min_n=self.min_n, max_n=self.max_n)
        return self.lang_prof.predict(doc, preprocess, self.rank_counts)


class EnsembleLI:
    """Ensemble Language Identification"""
    def __init__(self, k=300, min_n=1, max_n=5):
        """
        Initialize the 3 individual language identification models
        Parameters
        ----------
        k : int
            Number of top frequent words/n-grams to keep in each profile, 0 to keep everything
        min_n : int
            Minimum number of characters in the n-grams
        max_n : int
            Maximum number of characters in the n-grams
        """
        self.word = WordLI(k=k)
        self.range = NGramRangeLI(k=k, min_n=min_n, max_n=max_n)
        self.rank = RankNGramRangeLI(k=k, min_n=min_n, max_n=max_n)

    def train(self, df):
        """
        Train the individual models

        Parameters
        ----------
        df : DataFrame

        Returns
        -------

        """
        self.word.train(df)
        self.range.train(df)
        self.rank.train(df)

    def vote(self, predictions):
        """
        Voting classifier, counts each vote and returns the most common one. In the case of a tie chooses the first one
        """
        clf = Counter(predictions.to_list())
        return clf.most_common(1)[0][0]

    def predict(self, df):
        self.rank.predict(df)
        df.rename({'y_pred': 'rank_pred'}, axis=1, inplace=True)
        self.word.predict(df)
        df.rename({'y_pred': 'word_pred'}, axis=1, inplace=True)
        self.range.predict(df)
        df.rename({'y_pred': 'range_pred'}, axis=1, inplace=True)

        df['y_pred'] = df[['rank_pred', 'range_pred', 'word_pred']].apply(self.vote, axis=1)

    def predict_doc(self, doc):
        predictions = []
        predictions.append(self.word.predict_doc(doc))
        predictions.append(self.word.predict_doc(doc))
        predictions.append(self.word.predict_doc(doc))

        return Counter(predictions).most_common(1)[0][0]

    def accuracy(self, df):

        return self.word.accuracy(df)

    def lang_accuracy(self, df):

        return self.word.lang_accuracy(df)

    def to_json(self):
        self.word.to_json(filename='word_lp.json')
        self.range.to_json(filename='range_lp.json')
        self.rank.to_json(filename='rank_lp.json')

    def load(self):
        self.word.load(filename='word_lp.json')
        self.range.load(filename='range_lp.json')
        self.rank.load(filename='rank_lp.json')
