from language_modelling.corpus import Corpus, ReutersTrainingCorpus
from language_modelling.training.interpolation_trainer import InterpolationTrainer
from language_modelling.perplexity import PerplexityCalculator
from language_modelling.model.ngram_language_model import NgramLanguageModel
from language_modelling.model.ngram_probability_calculator import InterpolatingNgramProbabilityCalculator
from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer


class TestInterpolationTrainer:
    def test_trainer_on_reuters_corpus(self):
        reuters = ReutersTrainingCorpus().get_sentences()
        slice_size = round(len(reuters) * .1)

        guesses = []

        for i in range(10):
            slice_index = round(slice_size * i)
            slice_end = slice_index + slice_size
            training = Corpus(reuters[:slice_index] + reuters[slice_end:])
            holdout = Corpus(reuters[slice_index:slice_end])
            best_lambdas = InterpolationTrainer().find_lambdas_max_estimation(training, holdout, 3)
            guesses.append(best_lambdas)

        perplexities = [0] * 10
        for slicing in range(10):
            slice_index = round(slice_size * i)
            slice_end = slice_index + slice_size
            training = Corpus(reuters[:slice_index] + reuters[slice_end:])
            holdout = Corpus(reuters[slice_index:slice_end])
            for guess in range(10):
                calculator = InterpolatingNgramProbabilityCalculator(NgramCalculatorContainer(training, 3), guesses[guess])
                max_estimation_language_model = NgramLanguageModel(training, 3, ngram_probability_calculator=calculator)
                perplexity = PerplexityCalculator().calculate_corpus_perplexity(max_estimation_language_model, holdout)
                perplexities[guess] += perplexity

        min_perplexity = min(perplexities)
        best_lambdas = guesses[perplexities.index(min_perplexity)]

        for i in range(10):
            print(perplexities[i] / 10, guesses[i])

        print(min_perplexity, best_lambdas)
