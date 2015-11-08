from language_modelling.out_of_vocabulary_rate_calculator import OutOfVocabularyRateCalculator
from language_modelling.corpus import Corpus


class TestOutOfVocabularyRateCalculator:
    training_corpus = [
        ['how', 'many', 'roads', 'must', 'a', 'man', 'walk', 'down'],
        ['before', 'you', 'call', 'him', 'a', 'man'],
        ['how', 'many', 'seas', 'must', 'a', 'white', 'dove', 'sail'],
        ['before', 'she', 'sleeps', 'in', 'the', 'sand']
    ]
    test_corpus = [
        ['yes', 'and', 'how', 'many', 'times', 'must', 'the', 'cannon', 'balls', 'fly'],
        ['before', "they're", 'forever', 'banned'],
        ['the', 'answer', 'my', 'friend', 'is', "blowin'", 'in', 'the', 'wind'],
        ['the', 'answer', 'is', "blowin'", 'in', 'the', 'wind']
    ]

    calculator = OutOfVocabularyRateCalculator()

    def test_calculate_out_of_vocabulary_rate(self):
        assert 19/30*100 == self.calculator.calculate_out_of_vocabulary_rate(Corpus(self.training_corpus),
                                                                             Corpus(self.test_corpus))
