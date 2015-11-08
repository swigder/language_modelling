
class PerplexityCalculator:
    @staticmethod
    def calculate_corpus_perplexity(language_model, corpus):
        entropy = 0
        words = 0
        for sentence in corpus.get_sentences():
            sentence_entropy, sentence_found_words = language_model.get_sentence_log_probability(sentence)
            entropy += sentence_entropy
            words += sentence_found_words
        entropy /= -words
        perplexity = 2 ** entropy
        return perplexity
