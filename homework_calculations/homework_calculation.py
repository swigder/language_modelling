from language_modelling.reuters_training_corpus import ReutersTrainingCorpus
from language_modelling.unigram_calculator import UnigramCalculator
from language_modelling.ngram_calculator import NgramCalculator


def q2_calculate_unique_unigrams_in_reuters_training():
    corpus = ReutersTrainingCorpus()
    unigram_calculator = UnigramCalculator(corpus)
    ngram_calculator = NgramCalculator(corpus)

    unique_unigrams = unigram_calculator.get_percentage_unique_unigrams()
    unique_unigrams_ngram = ngram_calculator.get_percentage_unique_ngrams(1)

    print("Percentage of unique unigrams (unigram calculator): ", unique_unigrams)
    print("Percentage of unique unigrams (ngram calculator): ", unique_unigrams_ngram)


def q3_calculate_unique_ngrams_in_reuters_training_for_n_2_3_4_5_6():
    corpus = ReutersTrainingCorpus()
    ngram_calculator = NgramCalculator(corpus)

    unique_ngrams_2 = ngram_calculator.get_percentage_unique_ngrams(2)
    unique_ngrams_3 = ngram_calculator.get_percentage_unique_ngrams(3)
    unique_ngrams_4 = ngram_calculator.get_percentage_unique_ngrams(4)
    unique_ngrams_5 = ngram_calculator.get_percentage_unique_ngrams(5)
    unique_ngrams_6 = ngram_calculator.get_percentage_unique_ngrams(6)

    print("Percentage of unique bigrams: ", unique_ngrams_2)
    print("Percentage of unique trigrams: ", unique_ngrams_3)
    print("Percentage of unique 4-grams: ", unique_ngrams_4)
    print("Percentage of unique 5-grams: ", unique_ngrams_5)
    print("Percentage of unique 6-grams: ", unique_ngrams_6)


if __name__ == '__main__':
    q2_calculate_unique_unigrams_in_reuters_training()
    q3_calculate_unique_ngrams_in_reuters_training_for_n_2_3_4_5_6()
