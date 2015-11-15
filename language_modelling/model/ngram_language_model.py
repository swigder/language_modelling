from math import log

from language_modelling.model.ngram_calculator_container import NgramCalculatorContainer


class NgramLanguageModel:
    """
    Language model using ngram probabilities
    """

    BASE = 2

    def __init__(self, corpus, n, calculator_class=None, ngram_counter=None, ngram_probability_calculator=None):
        """
        :param corpus: corpus over which to create the language model
        """
        self.n = n
        self.ngram_counter = NgramCalculatorContainer(corpus, n) if ngram_counter is not None else ngram_counter
        self.ngram_probability_calculator = calculator_class(self.ngram_counter) \
            if calculator_class is not None \
            else ngram_probability_calculator

    def get_ngram_probability(self, ngram):
        """
        Gets the probability of a ngram as calculated by the configured ngram_probability_calculator
        :param ngram: ngram to get the probability for
        :return: probability of the ngram for the language model
        """
        return self.ngram_probability_calculator.get_ngram_probability(ngram)

    def get_ngram_log_probability(self, ngram):
        """
        Gets the log of the probability of a bigram; see get_bigram_probability for details
        :param ngram: ngram to get the probability for
        :return: log base 2 of the probability of the bigram for the language model; None if bigram probability is 0
        """
        n_probability = self.get_ngram_probability(ngram)
        return log(n_probability, self.BASE) if n_probability != 0 else None

    def get_sentence_log_probability(self, sentence):
        """
        Calculates the log probability of a sentence
        :param sentence: list of words
        :return: log of the probability of the sentence (to be used to calculate entropy, perplexity)
        """
        probability = 0
        found_words = 0
        n = self.n
        padding = ['<s>'] * (n-1)
        sentence = padding + sentence
        for i in range(n-1, len(sentence)):
            ngram = sentence[i-n+1:i+1]
            ngram_probability = self.get_ngram_log_probability(ngram)
            probability += ngram_probability if ngram_probability is not None else 0
            found_words += 1 if ngram_probability is not None else 0
        return probability, found_words
