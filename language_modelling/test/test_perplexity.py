from language_modelling.corpus import Corpus, BrownCorpus
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.bigram_language_model import UnigramLanguageModel, BigramLanguageModel


class TestPerplexityCalculator:

    calculator = PerplexityCalculator()

    def test_very_simple_perplexity(self):
        corpus = Corpus([['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']])
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)

        assert 10 == round(self.calculator.calculate_corpus_perplexity(unigram_language_model, corpus), 10)
        assert 1 == round(self.calculator.calculate_corpus_perplexity(bigram_language_model, corpus), 10)

    def test_calculate_corpus_perplexity(self):
        corpus = BrownCorpus()
        unigram_language_model = UnigramLanguageModel(corpus)
        bigram_language_model = BigramLanguageModel(corpus)

        unigram_perplexity = self.calculator.calculate_corpus_perplexity(unigram_language_model, corpus)
        bigram_perplexity = self.calculator.calculate_corpus_perplexity(bigram_language_model, corpus)
        print(unigram_perplexity, bigram_perplexity)
