

class NgramProbabilityCalculator:

    def __init__(self, ngram_counter):
        self.ngram_counter = ngram_counter
        self.corpus_unigram_length = ngram_counter.corpus_unigram_length
        self.corpus_sentence_length = ngram_counter.corpus_unigram_length
        self.corpus_vocabulary_size = len(ngram_counter.get_pregram_instances([]))

    def get_ngram_probability(self, ngram):
        ngram_count = self.ngram_counter.get_ngram_count(ngram)

        if ngram_count == 0:
            return 0

        if len(ngram) == 1:
            return ngram_count / self.corpus_unigram_length

        pregram = ngram[:-1]
        pregram_count = self.ngram_counter.get_ngram_count(pregram)

        return ngram_count / pregram_count


class LaplaceSmoothingNgramProbabilityCalculator(NgramProbabilityCalculator):

    def __init__(self, ngram_counter):
        super().__init__(ngram_counter)

    def get_ngram_probability(self, ngram):
        ngram_count = self.ngram_counter.get_ngram_count(ngram)

        pregram = ngram[:-1]
        pregram_count = self.ngram_counter.get_ngram_count(pregram)

        return (ngram_count + 1) / (pregram_count + self.corpus_vocabulary_size)


class InterpolatingNgramProbabilityCalculator(NgramProbabilityCalculator):

    def __init__(self, ngram_counter, lambdas=None):
        super(InterpolatingNgramProbabilityCalculator, self).__init__(ngram_counter)
        self.n = ngram_counter.n
        self.lambdas = lambdas if lambdas is not None else [1/self.n] * self.n

    def get_ngram_probability(self, ngram):
        probability = 0
        for i in range(self.n):
            probability += self.lambdas[i] * super().get_ngram_probability(ngram[i:])
        return probability


class BackoffNgramProbabilityCalculator(NgramProbabilityCalculator):

    def __init__(self, ngram_counter):
        super().__init__(ngram_counter)

    def get_ngram_probability(self, ngram):
        ngram_probability = super().get_ngram_probability(ngram)

        if ngram_probability == 0 and len(ngram) > 1:
            return 0.4 * self.get_ngram_probability(ngram[1:])

        return ngram_probability

