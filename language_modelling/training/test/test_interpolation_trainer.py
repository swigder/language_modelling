from language_modelling.corpus import Corpus, ReutersTrainingCorpus
from language_modelling.training.interpolation_trainer import InterpolationTrainer
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_probability_calculator import InterpolatingNgramProbabilityCalculator
from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer


class TestInterpolationTrainer:
    def test_trainer_on_reuters_corpus(self):
        reuters = ReutersTrainingCorpus().get_sentences()
        slice_index = round(len(reuters) * .9)
        training = Corpus(reuters[:slice_index])
        holdout = Corpus(reuters[slice_index:])

        best_lambdas, minimum_perplexity = InterpolationTrainer().find_lambdas_brute_force(training, holdout, 3)
        max_estimation_guess = [0.25152437827417695, 0.469454648600852, 0.2790209731249711]
        calculator = InterpolatingNgramProbabilityCalculator(NgramCalculatorContainer(training, 3), max_estimation_guess)
        max_estimation_language_model = NgramLanguageModel(training, 3, ngram_probability_calculator=calculator)
        max_estimation_perplexity = PerplexityCalculator().calculate_corpus_perplexity(max_estimation_language_model, holdout)

        print(best_lambdas, minimum_perplexity)
        print(max_estimation_guess, max_estimation_perplexity)
