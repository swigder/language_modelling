
class PerplexityCalculator:
    @staticmethod
    def calculate_corpus_perplexity(language_model, corpus):
        entropy = 0
        words = 0
        for sentence in corpus.get_sentences():
            words += len(sentence)
            entropy += language_model.get_sentence_log_probability(sentence)
        entropy /= -words
        perplexity = 2 ** entropy
        return perplexity
