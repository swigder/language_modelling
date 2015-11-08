from language_modelling.unigram_calculator import UnigramCalculator


class Corpus:
    sentences = [
            ['this', 'is', 'a', 'sentence'],
            ['a', 'word', 'is', 'small'],
            ['must', 'think', 'of', 'some', 'more', 'data'],
            ['let', 'us', 'get', 'to', 'twenty', 'words']
        ]

    unigrams = {
        'this': 1,
        'is': 2,
        'a': 2,
        'sentence': 1,
        'word': 1,
        'small': 1,
        'must': 1,
        'think': 1,
        'of': 1,
        'some': 1,
        'more': 1,
        'data': 1,
        'let': 1,
        'us': 1,
        'get': 1,
        'to': 1,
        'twenty': 1,
        'words': 1
    }

    def get_sentences(self):
        return self.sentences


class TestUnigramCalculator:
    corpus = Corpus()
    unigram_calculator = UnigramCalculator(corpus)

    def test_calculate_unigrams(self):
        assert 20 == self.unigram_calculator.total_unigrams
        assert self.corpus.unigrams == self.unigram_calculator.unigrams

    def test_get_percentage_unique_unigrams(self):
        assert 16/18 == self.unigram_calculator.get_percentage_unique_unigrams()