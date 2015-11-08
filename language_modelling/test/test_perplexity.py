from language_modelling.corpus import BrownCorpus
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.bigram_language_model import BigramLanguageModel


class TestPerplexityCalculator:

    corpus = BrownCorpus()
    language_model = BigramLanguageModel(corpus)
    calculator = PerplexityCalculator()

    def test_calculate_corpus_perplexity(self):
        perplexity = self.calculator.calculate_corpus_perplexity(self.language_model, self.corpus)
        print(perplexity)


