class NgramCalculator:
    """
    Calculates the ngrams in a corpus and provides data about uniqueness among ngrams
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def calculate_ngrams(self, n, pad_left=False, pad_right=False):
        """
        Calculate the ngrams in a corpus
        :param n: tokens in the ngrams
        :param pad_left: whether to pad the beginning of each sentence with sentence-boundary tokens
        :param pad_right: whether to pad the end of each sentence with sentence-boundary tokens
        :return: dictionary of ngrams to their count in the corpus
        """
        ngrams = {}
        padding = ['<s>'] * (n-1)
        for sentence in self.corpus.get_sentences():
            if pad_left:
                sentence = padding + sentence
            if pad_right:
                sentence = sentence + padding
            for i in range(len(sentence)-n+1):
                ngram = sentence[i:i+n+1]
                ngram = ngram[0] if n == 1 else tuple(ngram)
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1
        return ngrams

    def get_unique_ngrams(self, n):
        """
        Calculates unique ngrams in a corpus
        :param n: tokens in the ngrams
        :return: list of ngrams that appear only once in the corpus
        """
        unique_ngrams = []
        for ngram, instances in self.calculate_ngrams(n).items():
            if instances == 1:
                unique_ngrams.append(ngram)
        return unique_ngrams

    def get_percentage_unique_ngrams(self, n):
        """
        Calculate percentage of unique ngrams in the corpus.
        This method does more than it should because it calculates all the ngrams twice.  Should be refactored.
        :param n: tokens in the ngrams
        :return: proportion of unique ngrams that appear only once; i.e., number of ngrams that appear only once over
        total number of unique ngrams.
        """
        return len(self.get_unique_ngrams(n)) / len(self.calculate_ngrams(n)) * 100
