import sys
import argparse
from models import EnsembleLI



def main():
    parser = argparse.ArgumentParser(description='Predicts the language of text')

    parser.add_argument('--k', type=int, default=300,
                        help="Limits the size of the language profiles to only the top-k most frequent ones (default: "
                             "%(default)s)")
    parser.add_argument('--min_n', type=int, default=1,
                        help="Minimum number of characters in the n-grams (default: %(default)s)")
    parser.add_argument('--max_n', type=int, default=7,
                        help="Maximum number of characters in the n-grams (default: %(default)s)")

    args = parser.parse_args()

    model = EnsembleLI(k=args.k, min_n=args.min_n, max_n=args.max_n)
    model.load()
    doc = sys.stdin.readline()
    print(model.predict_doc(doc))


if __name__ == '__main__':
    main()
