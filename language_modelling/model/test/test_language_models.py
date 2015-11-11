from language_modelling.corpus import Corpus, BrownCorpus, ReutersTrainingCorpus
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.bigram_language_model import UnigramLanguageModel, BigramLanguageModel
from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_probability_calculator import NgramProbabilityCalculator, \
    BackoffNgramProbabilityCalculator, LaplaceSmoothingNgramProbabilityCalculator


class TestLanguageModels:

    calculator = PerplexityCalculator()

    def test_language_models_on_very_simple_corpus(self):
        corpus = Corpus([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']])
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)
        trigram_language_model = NgramLanguageModel(corpus, 3, NgramProbabilityCalculator)

        assert 10 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, corpus), 10)
        assert 1 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, corpus), 10)
        assert 1 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, corpus), 10)

    def test_language_models_on_somewhat_simple_corpus(self):
        training = Corpus([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']])
        test = Corpus([['a', 'b', 'c'], ['a', 'c', 'd']])
        unigram_language_model = UnigramLanguageModel(training)
        bigram_language_model = BigramLanguageModel(training)
        bigram_language_model_smoothing = NgramLanguageModel(training, 2, LaplaceSmoothingNgramProbabilityCalculator)
        trigram_language_model = NgramLanguageModel(training, 3, NgramProbabilityCalculator)
        trigram_language_model_backoff = NgramLanguageModel(training, 3, BackoffNgramProbabilityCalculator)

        assert 4.95 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, test), 2)
        assert 1.68 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, test), 2)
        assert 3.49 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model_smoothing, test), 2)
        assert 1.73 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, test), 2)
        assert 1.85 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model_backoff, test), 2)

    def test_language_models_on_brown_corpus(self):
        corpus = BrownCorpus()
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)
        trigram_language_model = NgramLanguageModel(corpus, 3, BackoffNgramProbabilityCalculator)
        quadrigram_language_model = NgramLanguageModel(corpus, 4, BackoffNgramProbabilityCalculator)

        assert 1548.66 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, corpus), 2)
        assert 100.10 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, corpus), 2)
        assert 7.77 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, corpus), 2)
        assert 2.40 == round(self.calculator.calculate_corpus_perplexity(quadrigram_language_model, corpus), 2)

    def test_language_models_on_reuters_corpus(self):
        reuters = ReutersTrainingCorpus().get_sentences()
        slice_index = round(len(reuters) * .9)
        training_corpus = Corpus(reuters[:slice_index])
        test_corpus = Corpus(reuters[slice_index:])

        unigram_language_model = UnigramLanguageModel(training_corpus)
        bigram_language_model = NgramLanguageModel(training_corpus, 2, BackoffNgramProbabilityCalculator)
        trigram_language_model = NgramLanguageModel(training_corpus, 3, BackoffNgramProbabilityCalculator)
        quadrigram_language_model = NgramLanguageModel(training_corpus, 4, BackoffNgramProbabilityCalculator)

        assert 1021.63 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, test_corpus), 2)
        assert 145.54 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, test_corpus), 2)
        assert 86.26 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, test_corpus), 2)
        assert 75.61 == round(self.calculator.calculate_corpus_perplexity(quadrigram_language_model, test_corpus), 2)
