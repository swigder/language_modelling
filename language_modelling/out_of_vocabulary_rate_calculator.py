class OutOfVocabularyRateCalculator:
    """
    Calculates the out-of-vocabulary rate of one corpus with respect to another.
    """

    @staticmethod
    def calculate_out_of_vocabulary_rate(training_corpus, test_corpus):
        """
        Calculate the out-of-vocabulary rate of a test corpus with respect to the training corpus
        :param training_corpus: training data, in format list of list of tokens
        :param test_corpus: test data, in format list of list of tokens
        :return: percentage of tokens in test corpus which do not appear in training corpus
        """
        training_vocabulary = set([item for sublist in training_corpus.get_sentences() for item in sublist])
        test_tokens = [item for sublist in test_corpus.get_sentences() for item in sublist]

        out_of_vocabulary_words = 0
        for token in test_tokens:
            if token not in training_vocabulary:
                out_of_vocabulary_words += 1

        return (out_of_vocabulary_words / len(test_tokens))
