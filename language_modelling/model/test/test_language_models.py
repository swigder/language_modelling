from language_modelling.corpus import Corpus, BrownCorpus, ReutersTrainingCorpus
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.bigram_language_model import UnigramLanguageModel, BigramLanguageModel
from language_modelling.model.ngram_language_model import NgramLanguageModel


class TestLanguageModels:

    calculator = PerplexityCalculator()

    def test_language_models_on_very_simple_corpus(self):
        corpus = Corpus([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']])
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)
        trigram_language_model = NgramLanguageModel(corpus, 3)

        assert 10 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, corpus), 10)
        assert 1 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, corpus), 10)
        assert 1 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, corpus), 10)

    def test_language_models_on_brown_corpus(self):
        corpus = BrownCorpus()
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)
        trigram_language_model = NgramLanguageModel(corpus, 3)
        quadrigram_language_model = NgramLanguageModel(corpus, 4)

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
        bigram_language_model = NgramLanguageModel(training_corpus, 2)
        trigram_language_model = NgramLanguageModel(training_corpus, 3)
        quadrigram_language_model = NgramLanguageModel(training_corpus, 4)

        assert 1021.63 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, test_corpus), 2)
        assert 74.20 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, test_corpus), 2)
        assert 86.26 == round(self.calculator.calculate_corpus_perplexity(trigram_language_model, test_corpus), 2)
        assert 75.61 == round(self.calculator.calculate_corpus_perplexity(quadrigram_language_model, test_corpus), 2)
