from language_modelling.corpus import ReutersTrainingCorpus, ReutersTestCorpus
from language_modelling.unigram_calculator import UnigramCalculator
from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.out_of_vocabulary_rate_calculator import OutOfVocabularyRateCalculator
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.bigram_language_model import BigramLanguageModel
from language_modelling.model.ngram_language_model import NgramLanguageModel


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


def q4_calculate_oov_rate_for_reuters_test_with_respect_to_training():
    training = ReutersTrainingCorpus()
    test = ReutersTestCorpus()
    oov_calculator = OutOfVocabularyRateCalculator()

    oov_rate = oov_calculator.calculate_out_of_vocabulary_rate(training, test)
    print("Out of vocabulary rate: ", oov_rate)


def q5_calculate_perplexity_of_reuters_corpus():
    training = ReutersTrainingCorpus()
    test = ReutersTestCorpus()
    language_model_bigram = BigramLanguageModel(training)
    language_model_ngram = NgramLanguageModel(training, 2, backoff=False)
    perplexity_calculator = PerplexityCalculator()
    training_perplexity_bigram = perplexity_calculator.calculate_corpus_perplexity(language_model_bigram, training)
    test_perplexity_bigram = perplexity_calculator.calculate_corpus_perplexity(language_model_bigram, test)
    training_perplexity_ngram = perplexity_calculator.calculate_corpus_perplexity(language_model_ngram, training)
    test_perplexity_ngram = perplexity_calculator.calculate_corpus_perplexity(language_model_ngram, test)

    print("Training perplexity: ", training_perplexity_bigram, training_perplexity_ngram)
    print("Test perplexity: ", test_perplexity_bigram, test_perplexity_ngram)


def q6_minimize_perplexity_of_reuters_corpus():
    training = ReutersTrainingCorpus()
    test = ReutersTestCorpus()
    language_model = BigramLanguageModel(training)
    perplexity_calculator = PerplexityCalculator()
    bigram_perplexity = perplexity_calculator.calculate_corpus_perplexity(language_model, test)

    print("Bigram perplexity: ", bigram_perplexity)


if __name__ == '__main__':
    q2_calculate_unique_unigrams_in_reuters_training()
    q3_calculate_unique_ngrams_in_reuters_training_for_n_2_3_4_5_6()
    q4_calculate_oov_rate_for_reuters_test_with_respect_to_training()
    q5_calculate_perplexity_of_reuters_corpus()
