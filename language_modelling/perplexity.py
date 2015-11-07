from math import pow
from math import log
from math import exp


class PerplexityCalculator:
    @staticmethod
    def calculate_sentence_probability(language_model, sentence):
        probability = log(language_model.get_probability(('<s>', sentence[0])))
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            probability += log(language_model.get_probability(bigram))
        return probability

    def calculate_corpus_perplexity(self, language_model, corpus):
        entropy = 0
        words = 0
        for sentence in corpus.get_sentences():
            words += len(sentence)
            entropy += self.calculate_sentence_probability(language_model, sentence)
        entropy *= -1/words
        perplexity = exp(entropy)
        return perplexity
