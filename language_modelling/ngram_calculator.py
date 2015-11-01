class NgramCalculator:

    def __init__(self, corpus):
        self.corpus = corpus

    def calculate_ngrams(self, n, pad_left = False, pad_right = False):
        ngrams = {}
        padding = ['<s>'] * (n-1)
        for sentence in self.corpus.get_sentences():
            if pad_left:
                sentence = padding + sentence
            if pad_right:
                sentence = sentence + padding
            for i in range(len(sentence)-n+1):
                ngram = sentence[i:i+n+1]
                ngram = ngram[0] if n == 1 else tuple(ngram)
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1
        return ngrams

    def get_unique_ngrams(self, n):
        unique_ngrams = []
        for ngram, instances in self.calculate_ngrams(n).items():
            if instances == 1:
                unique_ngrams.append(ngram)
        return unique_ngrams

    def get_percentage_unique_ngrams(self, n):
        return len(self.get_unique_ngrams(n)) / len(self.calculate_ngrams(n)) * 100
