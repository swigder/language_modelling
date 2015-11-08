from language_modelling.unigram_calculator import UnigramCalculator
from math import log


class UnigramLanguageModel:
    """
    Language model using only unigram probabilities
    """

    BASE = 2

    def __init__(self, corpus):
        """
        :param corpus: corpus over which to create the language model
        """
        unigram_calculator = UnigramCalculator(corpus)
        self.unigrams, self.corpus_unigram_length = unigram_calculator.calculate_unigrams()
        self.corpus_sentence_length = len(corpus.get_sentences())

    def get_unigram_count(self, unigram):
        """
        Get the count of a unigram in the corpus
        :param unigram: unigram to find count for
        :return: count in the corpus if the unigram is found,
        1 if it is not found, TODO should be zero
        number of sentences in the corpus if the sentence boundary marker, <s>, is provided
        """
        if unigram == '<s>':
            return self.corpus_sentence_length
        if unigram in self.unigrams:
            return self.unigrams[unigram]
        return 1  # don't want to cause math error by returning 0; numerator will be zero anyway

    def get_sentence_log_probability(self, sentence):
        """
        Calculates the log probability of a sentence
        :param sentence: list of words
        :return: log of the probability of the sentence (to be used to calculate entropy, perplexity)
        """
        probability = 0
        for i in range(0, len(sentence)):
            probability += log(self.get_unigram_count(sentence[i]) / self.corpus_unigram_length, self.BASE)
        return probability

    def get_sentence_probability(self, sentence):
        """
        Calculates the probability of a sentence
        :param sentence: list of words
        :return: probability of the sentence
        """
        return self.BASE ** self.get_sentence_log_probability(sentence)
