from nltk.corpus import reuters


class ReutersTrainingCorpus:
    """
    Wrapper around the Reuters Corpus that returns only training data.
    """

    training_files = [fileid for fileid in reuters.fileids() if fileid.startswith('training')]
    sentences = reuters.sents(training_files)

    def get_sentences(self):
        """
        Get all the sentences in the Reuters corpus which are part of the training data
        :return: list of lists of tokens, where each sublist is a sentence in the corpus
        """
        return self.sentences
