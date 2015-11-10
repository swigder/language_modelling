from math import log

from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.model.unigram_language_model import UnigramLanguageModel


class BigramLanguageModel(UnigramLanguageModel):
    """
    Language model using bigram probabilities
    """

    def __init__(self, corpus):
        """
        :param corpus: corpus over which to create the language model
        """
        super(BigramLanguageModel, self).__init__(corpus)
        self.ngram_calculator = NgramCalculator(corpus)
        self.bigrams = self.ngram_calculator.calculate_ngrams(2, True, False)

    def get_bigram_count(self, bigram):
        """
        Gets the count of the bigram in the corpus
        :param bigram: bigram to find in the corpus
        :return: number of times the bigram is found in the corpus; 1 if it is not found TODO should be zero
        """
        return self.bigrams[bigram] if bigram in self.bigrams else 0

    def get_bigram_probability(self, bigram):
        """
        Gets the probability of a bigram as calculated by count(bigram) / count(first word of bigram)
        :param bigram: bigram to get the probability for
        :return: probability of the bigram for the language model
        """
        x, y = bigram
        x_count = self.get_unigram_count(x)
        return self.get_bigram_count(bigram) / x_count if x_count != 0 else 0

    def get_bigram_log_probability(self, bigram):
        """
        Gets the log of the probability of a bigram; see get_bigram_probability for details
        :param bigram: bigram to get the probability for
        :return: log base 2 of the probability of the bigram for the language model; None if bigram probability is 0
        """
        bigram_probability = self.get_bigram_probability(bigram)
        return log(bigram_probability, self.BASE) if bigram_probability != 0 else None

    def get_sentence_log_probability(self, sentence):
        """
        Calculates the log probability of a sentence
        :param sentence: list of words
        :return: log of the probability of the sentence (to be used to calculate entropy, perplexity)
        """
        probability = 0
        found_words = 0
        bigram_probability = self.get_bigram_log_probability(('<s>', sentence[0]))
        probability += bigram_probability if bigram_probability is not None else 0
        found_words += 1 if bigram_probability is not None else 0
        for i in range(1, len(sentence)):
            bigram = (sentence[i-1], sentence[i])
            bigram_probability = self.get_bigram_log_probability(bigram)
            probability += bigram_probability if bigram_probability is not None else 0
            found_words += 1 if bigram_probability is not None else 0
        return probability, found_words
