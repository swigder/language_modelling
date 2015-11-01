from nltk.corpus import reuters


class ReutersTrainingCorpus:

    training_files = [fileid for fileid in reuters.fileids() if fileid.startswith('training')]
    sentences = reuters.sents(training_files)

    def get_sentences(self):
        return self.sentences
