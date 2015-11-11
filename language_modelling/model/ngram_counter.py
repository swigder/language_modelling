from language_modelling.ngram_calculator import NgramCalculator
from language_modelling.unigram_calculator import UnigramCalculator


class NgramCounter:
    def __init__(self, corpus, n):
        self.corpus_sentence_length = len(corpus.get_sentences())

        unigram_calculator = UnigramCalculator(corpus)
        self.unigrams, self.corpus_unigram_length = unigram_calculator.calculate_unigrams()

        ngram_calculator = NgramCalculator(corpus)

        self.ngrams = [self.unigrams]
        for i in range(2, n+1):
            self.ngrams.append(ngram_calculator.calculate_ngrams(i, True, False))

    def get_ngram_count(self, ngram):
        """
        Gets the count of the bigram in the corpus
        :param ngram: bigram to find in the corpus
        :return: number of times the bigram is found in the corpus
        """
        if ngram[0] == '<s>' and ngram[-1] == '<s>':  # special case where pregram can be <s>
            return self.corpus_sentence_length

        ngrams = self.ngrams[len(ngram)-1]
        if len(ngram) == 1:  # base case
            unigram_count = self.unigrams[ngram[0]] if ngram[0] in self.unigrams else 0
            return unigram_count

        tuple_ngram = tuple(ngram)
        if tuple_ngram in ngrams:  # best case
            return ngrams[tuple_ngram]

        return 0
