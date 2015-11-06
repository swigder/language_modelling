from math import pow


class PerplexityCalculator:
    @staticmethod
    def calculate_probability(language_model, sentence):
        probability = 1
        for i in range(len(sentence)):
            bigram = (sentence[i], sentence[i+1])
            probability *= language_model.get_probability(bigram)
        return probability

    def calculate_sentence_perplexity(self, language_model, sentence):
        return pow(self.calculate_probability(language_model, sentence), 1/len(sentence))

    def calculate_corpus_perplexity(self, language_model, corpus):
        perplexity = 1
        for sentence in corpus.get_sentences():
            perplexity *= self.calculate_sentence_perplexity(language_model, sentence)
        return perplexity
