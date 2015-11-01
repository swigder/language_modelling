from nltk.corpus import reuters


class ReutersTestCorpus:

    training_files = [fileid for fileid in reuters.fileids() if fileid.startswith('test')]
    sentences = reuters.sents(training_files)

    def get_sentences(self):
        return self.sentences
