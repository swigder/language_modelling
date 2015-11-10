from math import log

from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.model.unigram_language_model import UnigramLanguageModel


class NgramLanguageModel(UnigramLanguageModel):
    """
    Language model using ngram probabilities with backoff
    """

    def __init__(self, corpus, n, backoff=True):
        """
        :param corpus: corpus over which to create the language model
        """
        super(NgramLanguageModel, self).__init__(corpus)
        self.n = n
        self.backoff = backoff
        self.ngram_calculator = NgramCalculator(corpus)
        self.ngrams = [self.unigrams]
        for i in range(2, n+1):
            self.ngrams.append(self.ngram_calculator.calculate_ngrams(i, True, False))

    def get_ngram_count(self, ngram):
        """
        Gets the count of the bigram in the corpus
        :param bigram: bigram to find in the corpus
        :return: number of times the bigram is found in the corpus; 1 if it is not found TODO should be zero
        """
        if ngram[0] == '<s>' and ngram[-1] == '<s>':  # special case where pregram can be <s>
            return self.corpus_sentence_length, len(ngram)

        ngrams = self.ngrams[len(ngram)-1]
        if len(ngram) == 1:  # base case
            unigram_count = self.unigrams[ngram[0]] if ngram[0] in self.unigrams else 0
            return unigram_count, len(ngram)

        tuple_ngram = tuple(ngram)
        if tuple_ngram in ngrams:  # best case
            return ngrams[tuple_ngram], len(ngram)

        return self.get_ngram_count(ngram[1:]) if self.backoff else (0, len(ngram))

    def get_ngram_probability(self, ngram):
        """
        Gets the probability of a bigram as calculated by count(bigram) / count(first word of bigram)
        :param ngram: ngram to get the probability for
        :return: probability of the bigram for the language model
        """
        ngram_count, found_n = self.get_ngram_count(ngram)

        if ngram_count == 0:
            return 0

        if found_n == 1:
            return ngram_count / self.corpus_unigram_length

        pregram = ngram[len(ngram)-found_n:-1]
        pregram_count, _ = self.get_ngram_count(pregram)

        return ngram_count / pregram_count

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
