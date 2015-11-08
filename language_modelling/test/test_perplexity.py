from language_modelling.corpus import BrownCorpus
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.ngram_language_model import UnigramLanguageModel, BigramLanguageModel


class TestPerplexityCalculator:

    corpus = BrownCorpus()
    unigram_language_model = UnigramLanguageModel(corpus)
    bigram_language_model = BigramLanguageModel(corpus)
    calculator = PerplexityCalculator()

    def test_calculate_corpus_perplexity(self):
        unigram_perplexity = self.calculator.calculate_corpus_perplexity(self.unigram_language_model, self.corpus)
        bigram_perplexity = self.calculator.calculate_corpus_perplexity(self.bigram_language_model, self.corpus)
        print(unigram_perplexity, bigram_perplexity)


