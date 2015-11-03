from nltk.corpus import brown


class BrownCorpus:
    """
    Wrapper around the Brown Corpus that caches sentences.
    """
    sentences = brown.sents()

    def get_sentences(self):
        """
        Get all the sentences in the Brown corpus
        :return: list of lists of tokens, where each sublist is a sentence in the corpus
        """
        return self.sentences
