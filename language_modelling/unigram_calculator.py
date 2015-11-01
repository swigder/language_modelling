class UnigramCalculator:

    def __init__(self, corpus):
        self.corpus = corpus
        self.unigrams, self.total_unigrams = self.calculate_unigrams()

    def calculate_unigrams(self):
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
        unique_unigrams = []
        for unigram, instances in self.unigrams.items():
            if instances == 1:
                unique_unigrams.append(unigram)
        return unique_unigrams

    def get_percentage_unique_unigrams(self):
        return len(self.get_unique_unigrams()) / len(self.unigrams) * 100
