from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.unigram_calculator import UnigramCalculator


class NgramCalculatorContainer:
    def __init__(self, corpus, n):
        self.n = n
        self.corpus_sentence_length = len(corpus.get_sentences())

        self.unigram_calculator = UnigramCalculator(corpus)
        self.corpus_unigram_length = self.unigram_calculator.total_unigrams

        self.ngram_calculators = []
        for i in range(2, n+1):
            self.ngram_calculators.append(NgramCalculator(corpus, i, True, False))

    def get_ngram_count(self, ngram):
        """
        Gets the count of the bigram in the corpus
        :param ngram: bigram to find in the corpus
        :return: number of times the bigram is found in the corpus
        """
        if ngram[0] == '<s>' and ngram[-1] == '<s>':  # special case where pregram can be <s>
            return self.corpus_sentence_length

        if len(ngram) == 1:  # base case
            unigram_count = self.unigram_calculator.get_ngram_count(ngram)
            return unigram_count

        ngram_calculator = self.ngram_calculators[len(ngram)-2]
        return ngram_calculator.get_ngram_count(ngram)

    def get_pregram_instances(self, pregram):
        if len(pregram) == 0:
            return self.unigram_calculator.get_pregram_instances(pregram)

        ngram_calculator = self.ngram_calculators[len(pregram)-1]
        return ngram_calculator.get_pregram_instances(pregram)
