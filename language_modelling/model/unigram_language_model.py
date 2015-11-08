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
        :return: count of the unigram in the corpus, or number of sentences in the corpus if the sentence boundary
        marker, <s>, is provided
        """
        if unigram == '<s>':
            return self.corpus_sentence_length
        if unigram in self.unigrams:
            return self.unigrams[unigram]
        return 0

    def get_sentence_log_probability(self, sentence):
        """
        Calculates the log probability of a sentence
        :param sentence: list of words
        :return: log of the probability of the sentence (to be used to calculate entropy, perplexity)
        """
        probability = 0
        words = 0
        for i in range(0, len(sentence)):
            unigram_count = self.get_unigram_count(sentence[i])
            probability += log(unigram_count / self.corpus_unigram_length, self.BASE) if unigram_count != 0 else 0
            words += 1 if unigram_count != 0 else 0
        return probability, words

    def get_sentence_probability(self, sentence):
        """
        Calculates the probability of a sentence
        :param sentence: list of words
        :return: probability of the sentence
        """
        sentence_probability, words = self.get_sentence_log_probability(sentence)
        return self.BASE ** sentence_probability, words
