from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer
from language_modelling.model.ngram_probability_calculator import InterpolatingNgramProbabilityCalculator
from language_modelling.perplexity import PerplexityCalculator
import sys


class InterpolationTrainer:

    def find_lambdas(self, training_corpus, holdout_corpus, n):
        ngram_counter = NgramCalculatorContainer(training_corpus, n)
        minimum_perplexity = sys.maxsize
        best_lambdas = []
        perplexity_calculator = PerplexityCalculator()
        increment = .1
        tries = round(1 / increment)

        for i in range(tries + 1):
            l1 = round(increment * i, 2)
            for j in range(tries-i):
                l2 = round(increment * j, 2)
                l3 = round(1 - l1 - l2, 2)
                lambdas = [l1, l2, l3]
                ngram_probability_calculator = InterpolatingNgramProbabilityCalculator(ngram_counter, lambdas)
                language_model = NgramLanguageModel(
                    training_corpus, n, ngram_probability_calculator=ngram_probability_calculator
                )
                perplexity = perplexity_calculator.calculate_corpus_perplexity(language_model, holdout_corpus)
                print(lambdas, perplexity)
                if perplexity < minimum_perplexity:
                    minimum_perplexity = perplexity
                    best_lambdas = lambdas

        return best_lambdas, minimum_perplexity
