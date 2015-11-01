from language_modelling.ngram_calculator import NgramCalculator
from nltk import FreqDist


class Corpus:
    sentences = [
            ['this', 'is', 'a', 'sentence'],
            ['a', 'word', 'is', 'small'],
            ['must', 'think', 'of', 'some', 'more', 'data'],
            ['let', 'us', 'get', 'to', 'twenty', 'words'],
            ['this', 'is', 'a', 'sentence', 'designed', 'to', 'get', 'some', 'longer', 'ngram', 'overlap'],
            ['i', 'would', 'like', 'some', 'more', 'to', 'drink'],
            ['do', 'you', 'have', 'the', 'time'],
            ['to', 'listen', 'to', 'me', 'whine'],
            ['do', 'not', 'say', 'anything', 'to', 'me']
        ]

    def get_sentences(self):
        return self.sentences


class TestNgramCalculator:
    corpus = Corpus()
    ngram_calculator = NgramCalculator(corpus)

    flattened_sentences = [item for sublist in corpus.get_sentences() for item in sublist]
    frequencies = FreqDist(flattened_sentences)

    def test_calculate_unigrams(self):
        assert dict(self.frequencies) == self.ngram_calculator.calculate_ngrams(1)

    def test_get_percentage_unique_unigrams(self):
        assert round(27/len(list(self.frequencies))*100, 2) == round(self.ngram_calculator.get_percentage_unique_ngrams(1), 2)
