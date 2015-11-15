from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer
from language_modelling.model.ngram_probability_calculator import InterpolatingNgramProbabilityCalculator
from language_modelling.model.ngram_probability_calculator import NgramProbabilityCalculator
from language_modelling.perplexity import PerplexityCalculator
import sys


class InterpolationTrainer:

    def find_lambdas_max_estimation(self, training_corpus, holdout_corpus, n):
        lambda_guesses = [1/n] * n
        ngram_counter = NgramCalculatorContainer(training_corpus, n)
        while True:
            summations = [0] * n
            calculator = NgramProbabilityCalculator(ngram_counter)
            interpolator = InterpolatingNgramProbabilityCalculator(ngram_counter, lambda_guesses)
            for sentence in holdout_corpus.get_sentences():
                sentence = ['<s>']*(n-1) + sentence
                for i in range(n-1, len(sentence)):
                    ngram = sentence[i-n+1:i+1]
                    lambda_estimation = interpolator.get_ngram_probability(ngram)
                    for j, lambda_i in enumerate(lambda_guesses):
                        sub_ngram = ngram[j:]
                        current_lambda_contribution = lambda_i * calculator.get_ngram_probability(sub_ngram)
                        summations[j] += current_lambda_contribution / lambda_estimation \
                            if lambda_estimation != 0 else 0
            summation_total = 0
            for summation in summations:
                summation_total += summation
            for i, lambda_i in enumerate(lambda_guesses):
                lambda_guesses[i] = summations[i] / summation_total
            print(lambda_guesses)

    def find_lambdas_brute_force(self, training_corpus, holdout_corpus, n):
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
                if perplexity < minimum_perplexity:
                    minimum_perplexity = perplexity
                    best_lambdas = lambdas

        return best_lambdas, minimum_perplexity
