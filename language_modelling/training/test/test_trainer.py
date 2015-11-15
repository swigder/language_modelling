from language_modelling.corpus import Corpus, ReutersTrainingCorpus
from language_modelling.training.interpolation_trainer import InterpolationTrainer


class TestTrainer:
    def test_trainer_on_reuters_corpus(self):
        reuters = ReutersTrainingCorpus().get_sentences()
        slice_index = round(len(reuters) * .9)
        training = Corpus(reuters[:slice_index])
        holdout = Corpus(reuters[slice_index:])

        best_lambdas, minimum_perplexity = InterpolationTrainer().find_lambdas(training, holdout, 3)

        print(best_lambdas, minimum_perplexity)
