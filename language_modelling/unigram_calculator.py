class UnigramCalculator:
    """
    Calculates the unigrams in a corpus and provides data about uniqueness among unigrams
    """

    def __init__(self, corpus):
        """
        :param corpus: the corpus for which to calculate unigrams
        """
        self.corpus = corpus
        self.unigrams, self.total_unigrams = self.calculate_unigrams()

    def calculate_unigrams(self):
        """
        Calculate the unigrams in the corpus
        :return: dictionary of unigrams that appear in the corpus to the number of times they appear
        """
        total_unigrams = 0
        unigrams = {}
        for sentence in self.corpus.get_sentences():
            for word in sentence:
                total_unigrams += 1
                if word in unigrams:
                    unigrams[word] += 1
                else:
                    unigrams[word] = 1

        return unigrams, total_unigrams

    def get_unique_unigrams(self):
        """
        Get the tokens which only appear once in the corpus
        :return: list of tokens which appear only once in the corpus
        """
        unique_unigrams = []
        for unigram, instances in self.unigrams.items():
            if instances == 1:
                unique_unigrams.append(unigram)
        return unique_unigrams

    def get_percentage_unique_unigrams(self):
        """
        Calculate percentage of unique tokens in the corpus.
        :return: proportion of tokens that appear only once; i.e., number of tokens that appear only once over total
        number of tokens.
        """
        return len(self.get_unique_unigrams()) / len(self.unigrams)
