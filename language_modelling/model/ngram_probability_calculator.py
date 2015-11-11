

class NgramProbabilityCalculator:

    def __init__(self, ngram_counter):
        self.ngram_counter = ngram_counter
        self.corpus_unigram_length = ngram_counter.corpus_unigram_length

    def get_ngram_probability(self, ngram):
        ngram_count = self.ngram_counter.get_ngram_count(ngram)

        if ngram_count == 0:
            return 0

        if len(ngram) == 1:
            return ngram_count / self.corpus_unigram_length

        pregram = ngram[:-1]
        pregram_count = self.ngram_counter.get_ngram_count(pregram)

        return ngram_count / pregram_count


class BackoffNgramProbabilityCalculator(NgramProbabilityCalculator):

    def __init__(self, ngram_counter):
        super(BackoffNgramProbabilityCalculator, self).__init__(ngram_counter)

    def get_ngram_probability(self, ngram):
        ngram_probability = super(BackoffNgramProbabilityCalculator, self).get_ngram_probability(ngram)

        if ngram_probability == 0 and len(ngram) > 1:
            return self.get_ngram_probability(ngram[1:])

        return ngram_probability

