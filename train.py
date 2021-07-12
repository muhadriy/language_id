import argparse
import pandas as pd
from models import EnsembleLI
from utils import load_data
import top10


def main():
    parser = argparse.ArgumentParser(description='Train an ensemble of three Language Identification models with a '
                                                 'hard voting classifier')
    parser.add_argument('--k', type=int, default=300,
                        help="Limits the size of the language profiles to only the top-k most frequent ones (default: "
                             "%(default)s)")
    parser.add_argument('--min_n', type=int, default=1,
                        help="Minimum number of characters in the n-grams (default: %(default)s)")
    parser.add_argument('--max_n', type=int, default=7,
                        help="Maximum number of characters in the n-grams (default: %(default)s)")
    parser.add_argument('--res', type=str, default="Results.csv",
                        help="Filename for the language specific and average accuracy (default: %(default)s)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lang_n', type=int, help="Number of languages to train on, randomly chosen")
    group.add_argument('--lang_p', type=float, help="Percentage of languages to train on, randomly chosen")
    group.add_argument('--top10', action='store_true', help="Train on the top 10 spoken languages in Europe")
    args = parser.parse_args()

    lang = None
    if args.lang_n:
        lang = args.lang_n

    if args.lang_p:
        lang = args.lang_p

    if args.top10:
        lang = top10.LANG_LIST

    train, _, test = load_data(languages=lang, dev_test_split=0.5)
    model = EnsembleLI(k=args.k, min_n=args.min_n, max_n=args.max_n)
    model.train(train)
    model.predict(test)
    model.to_json()

    lang_acc = model.lang_accuracy(test)
    lang_acc = pd.DataFrame.from_dict(lang_acc, orient='index', columns=['Accuracy'])
    lang_acc.loc['Average'] = lang_acc.mean()
    lang_acc.to_csv(args.res)


if __name__ == '__main__':
    main()
