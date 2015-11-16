from language_modelling.corpus import ReutersTrainingCorpus, ReutersTestCorpus, BrownCorpus
from language_modelling.unigram_calculator import UnigramCalculator
from language_modelling.basic_ngram_calculator import BasicNgramCalculator
from language_modelling.out_of_vocabulary_rate_calculator import OutOfVocabularyRateCalculator
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.bigram_language_model import BigramLanguageModel
from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_probability_calculator import InterpolatingNgramProbabilityCalculator
from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer


def q2_calculate_unique_unigrams_in_reuters_training():
    corpus = ReutersTrainingCorpus()
    unigram_calculator = UnigramCalculator(corpus)
    ngram_calculator = BasicNgramCalculator(corpus)

    unique_unigrams = unigram_calculator.get_percentage_unique_unigrams()
    unique_unigrams_ngram = ngram_calculator.get_percentage_unique_ngrams(1)

    print("Percentage of unique unigrams (unigram calculator): ", unique_unigrams)
    print("Percentage of unique unigrams (ngram calculator): ", unique_unigrams_ngram)


def q3_calculate_unique_ngrams_in_reuters_training_for_n_2_3_4_5_6():
    corpus = ReutersTrainingCorpus()
    ngram_calculator = BasicNgramCalculator(corpus)

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
    perplexity_calculator = PerplexityCalculator()
    training_perplexity_bigram = perplexity_calculator.calculate_corpus_perplexity(language_model_bigram, training)
    test_perplexity_bigram = perplexity_calculator.calculate_corpus_perplexity(language_model_bigram, test)

    print("Training perplexity basic model: ", training_perplexity_bigram)
    print("Test perplexity basic model: ", test_perplexity_bigram)


def q6_minimize_perplexity_of_reuters_corpus():
    training = ReutersTrainingCorpus()
    test = ReutersTestCorpus()
    lambdas = [0.2518, 0.4691, 0.2791]
    calculator = InterpolatingNgramProbabilityCalculator(NgramCalculatorContainer(training, 3), lambdas)
    language_model = NgramLanguageModel(training, 3, ngram_probability_calculator=calculator)
    perplexity = PerplexityCalculator().calculate_corpus_perplexity(language_model, test)

    print("Test perplexity best model: ", perplexity)


def q7_perplexity_oov_of_brown_corpus():
    training = ReutersTrainingCorpus()
    brown = BrownCorpus()
    lambdas = [0.2518, 0.4691, 0.2791]
    calculator = InterpolatingNgramProbabilityCalculator(NgramCalculatorContainer(training, 3), lambdas)
    language_model = NgramLanguageModel(training, 3, ngram_probability_calculator=calculator)
    perplexity = PerplexityCalculator().calculate_corpus_perplexity(language_model, brown)

    oov_calculator = OutOfVocabularyRateCalculator()
    oov_rate = oov_calculator.calculate_out_of_vocabulary_rate(training, brown)

    print("Brown perplexity best model: ", perplexity)
    print("Brown out of vocabulary rate: ", oov_rate)


if __name__ == '__main__':
    q2_calculate_unique_unigrams_in_reuters_training()
    q3_calculate_unique_ngrams_in_reuters_training_for_n_2_3_4_5_6()
    q4_calculate_oov_rate_for_reuters_test_with_respect_to_training()
    q5_calculate_perplexity_of_reuters_corpus()
    q6_minimize_perplexity_of_reuters_corpus()
    q7_perplexity_oov_of_brown_corpus()
