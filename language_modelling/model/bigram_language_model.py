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
        return self.bigrams[bigram] if bigram in self.bigrams else 1  # log(1) = 0 so adds 0 to sum

    def get_bigram_probability(self, bigram):
        """
        Gets the probability of a bigram as calculated by count(bigram) / count(first word of bigram)
        :param bigram: bigram to get the probability for
        :return: probability of the bigram for the language model
        """
        x, y = bigram
        return self.get_bigram_count(bigram) / self.get_unigram_count(x)

    def get_bigram_log_probability(self, bigram):
        """
        Gets the log of the probability of a bigram; see get_bigram_probability for details
        :param bigram: bigram to get the probability for
        :return: log base 2 of the probability of the bigram for the language model
        """
        return log(self.get_bigram_probability(bigram), 2)

    def get_sentence_log_probability(self, sentence):
        """
        Calculates the log probability of a sentence
        :param sentence: list of words
        :return: log of the probability of the sentence (to be used to calculate entropy, perplexity)
        """
        probability = self.get_bigram_log_probability(('<s>', sentence[0]))
        for i in range(1, len(sentence)):
            bigram = (sentence[i-1], sentence[i])
            probability += self.get_bigram_log_probability(bigram)
        return probability
