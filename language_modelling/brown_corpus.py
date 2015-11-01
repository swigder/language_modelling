from nltk.corpus import brown


class BrownCorpus:

    sentences = brown.sents()

    def get_sentences(self):
        return self.sentences
