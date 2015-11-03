class OutOfVocabularyRateCalculator:

    @staticmethod
    def calculate_out_of_vocabulary_rate(training_corpus, test_corpus):
        training_vocabulary = set([item for sublist in training_corpus for item in sublist])
        test_tokens = [item for sublist in test_corpus for item in sublist]

        out_of_vocabulary_words = 0
        for token in test_tokens:
            if token not in training_vocabulary:
                out_of_vocabulary_words += 1

        return out_of_vocabulary_words / len(test_tokens)
