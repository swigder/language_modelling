import re
from nltk.corpus import brown
from nltk.corpus import reuters


class Corpus:
    """
    Wrapper around a corpus that capitalizes sentences, removes punctuation, and caches content.
    """

    def __init__(self, unprocessed_sentences):
        sentences = [[word.upper() for word in sentence if len(re.sub(r'\W+', '', word)) != 0]
                     for sentence in unprocessed_sentences]
        self.sentences = [sentence for sentence in sentences if len(sentence) != 0]

    def get_sentences(self):
        """
        Get all the sentences in the Brown corpus
        :return: list of lists of tokens, where each sublist is a sentence in the corpus
        """
        return self.sentences


class BrownCorpus(Corpus):
    """
    Wrapper around Brown Corpus
    """
    def __init__(self):
        super(BrownCorpus, self).__init__(brown.sents())


class ReutersTestCorpus(Corpus):
    """
    Wrapper around the Reuters Corpus that returns only test data.
    """
    def __init__(self):
        test_files = [fileid for fileid in reuters.fileids() if fileid.startswith('test')]
        super(ReutersTestCorpus, self).__init__(reuters.sents(test_files))


class ReutersTrainingCorpus(Corpus):
    """
    Wrapper around the Reuters Corpus that returns only training data.
    """
    def __init__(self):
        training_files = [fileid for fileid in reuters.fileids() if fileid.startswith('training')]
        super(ReutersTrainingCorpus, self).__init__(reuters.sents(training_files))
