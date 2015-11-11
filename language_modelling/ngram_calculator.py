class NgramCalculator:
    """
    Calculates the ngrams in a corpus and provides data about uniqueness among ngrams
    """

    def __init__(self, corpus, n, pad_left=False, pad_right=False):
        self.ngrams = self.calculate_ngrams(corpus, n, pad_left, pad_right)

    def calculate_ngrams(self, corpus, n, pad_left=False, pad_right=False):
        """
        Calculate the ngrams in a corpus
        :param n: tokens in the ngrams
        :param pad_left: whether to pad the beginning of each sentence with sentence-boundary tokens
        :param pad_right: whether to pad the end of each sentence with sentence-boundary tokens
        :return: dictionary of ngrams to their count in the corpus
        """
        ngrams = {}
        padding = ['<s>'] * (n-1)
        for sentence in corpus.get_sentences():
            if pad_left:
                sentence = padding + sentence
            if pad_right:
                sentence = sentence + padding
            for i in range(len(sentence)-n+1):
                ngram_start = sentence[i:i+n-1]
                ngram_start = tuple(ngram_start)
                ngram_end = sentence[i+n-1]
                if ngram_start in ngrams:
                    ngram_start_dict = ngrams[ngram_start]
                    if ngram_end in ngram_start_dict:
                        ngram_start_dict[ngram_end] += 1
                    else:
                        ngram_start_dict[ngram_end] = 1
                else:
                    ngrams[ngram_start] = {ngram_end: 1}
        return ngrams

    def get_unique_ngrams(self):
        """
        Calculates unique ngrams in a corpus
        :param n: tokens in the ngrams
        :return: list of ngrams that appear only once in the corpus
        """
        unique_ngrams = []
        for ngram_start, ngram_ends in self.ngrams.items():
            if len(ngram_ends.items()) == 1:
                for ngram_end, instances in ngram_ends.items():
                    if instances == 1:
                        unique_ngrams.append(ngram_start + (ngram_end,))
        return unique_ngrams

    def get_percentage_unique_ngrams(self):
        """
        Calculate percentage of unique ngrams in the corpus.
        This method does more than it should because it calculates all the ngrams twice.  Should be refactored.
        :param n: tokens in the ngrams
        :return: proportion of unique ngrams that appear only once; i.e., number of ngrams that appear only once over
        total number of unique ngrams.
        """
        return len(self.get_unique_ngrams()) / len(self.ngrams)

    def get_ngram_count(self, ngram):
        """
        Get the number of times a given ngram appears in the training corpus
        :param ngram: ngram to find, as list
        :return: number of times the ngram appears in the corpus
        """
        pregram = tuple(ngram[:-1])
        final_word = ngram[-1]

        if pregram not in self.ngrams:
            return 0

        if final_word not in self.ngrams[pregram]:
            return 0

        return self.ngrams[pregram][final_word]

    def get_pregram_instances(self, pregram):
        pregram = tuple(pregram)
        if pregram not in self.ngrams:
            return None
        return self.ngrams[pregram]
