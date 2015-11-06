from language_modelling.ngram_calculator import NgramCalculator


class BigramLanguageModel:
    def __init__(self, corpus):
        ngram_calculator = NgramCalculator(corpus)
        self.unigrams = ngram_calculator.calculate_ngrams(1)
        self.bigrams = ngram_calculator.calculate_ngrams(2)

    def get_probability(self, bigram):
        x, y = bigram
        probability_y_given_x = self.bigrams[bigram]
        probability_x = self.unigrams[x]
        return probability_y_given_x / probability_x
