from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.unigram_calculator import UnigramCalculator
from math import log
from math import pow


class UnigramLanguageModel():
    def __init__(self, corpus):
        unigram_calculator = UnigramCalculator(corpus)
        self.unigrams, self.corpus_unigram_length = unigram_calculator.calculate_unigrams()
        self.corpus_sentence_length = len(corpus.get_sentences())

    def get_unigram_count(self, unigram):
        if unigram == '<s>':
            return self.corpus_sentence_length
        if unigram in self.unigrams:
            return self.unigrams[unigram]
        return 1  # don't want to cause math error by returning 0; numerator will be zero anyway

    def get_sentence_log_probability(self, sentence):
        probability = 0
        for i in range(0, len(sentence)):
            probability += log(self.get_unigram_count(sentence[i]) / self.corpus_unigram_length)
        return probability

    def get_sentence_probability(self, sentence):
        return pow(2, self.get_sentence_log_probability(sentence))


class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, corpus):
        super(BigramLanguageModel, self).__init__(corpus)
        self.ngram_calculator = NgramCalculator(corpus)
        self.bigrams = self.ngram_calculator.calculate_ngrams(2, True, False)

    def get_bigram_probability(self, bigram):
        x, y = bigram
        return self.get_bigram_count(bigram) / self.get_unigram_count(x)

    def get_bigram_count(self, bigram):
        return self.bigrams[bigram] if bigram in self.bigrams else 1  # log(1) = 0 so adds 0 to sum

    def get_bigram_log_probability(self, bigram):
        return log(self.get_bigram_probability(bigram), 2)

    def get_sentence_log_probability(self, sentence):
        probability = self.get_bigram_log_probability(('<s>', sentence[0]))
        for i in range(1, len(sentence)):
            bigram = (sentence[i-1], sentence[i])
            probability += self.get_bigram_log_probability(bigram)
        return probability
