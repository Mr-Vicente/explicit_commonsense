
#############################
#   Imports
#############################

# Python modules
import re
import string

# Remote modules

# Local modules
from utils import (
    read_simple_text_file_2_vec
)

#############################
#   Utils
#############################


class ParsingUtils:

    STOPWORDS = read_simple_text_file_2_vec('english_stopwords.txt', store_dir='kgs_binding')

    @staticmethod
    def remove_pontuation(text):
        text = re.sub(r"[^a-zA-Z]", " ", text)
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def clear_common_words(index_with_words):
        return [(word, (s, e)) for (word, (s, e)) in index_with_words if word not in ParsingUtils.STOPWORDS]

    @staticmethod
    def is_word_a_relevant_one(ignore_common_words, word):
        if ignore_common_words:
            return word not in ParsingUtils.STOPWORDS
        else:
            return True

    @staticmethod
    def get_word_range_mapping(context, word_token):
        word_token_splitted = word_token.split(' ')
        if len(word_token_splitted) == 1:
            word_token_start = context.index(word_token)
            word_token_end = word_token_start + len(word_token) - 1  # inclusive end
        else:
            word_token_start = context.index(word_token_splitted[0])
            word_token_end = word_token_start + len(word_token) - 1  # inclusive end
        return word_token_start, word_token_end

    @staticmethod
    def n_grams(words_vector, n):
        grams = [words_vector[i:i + n] for i in range(len(words_vector) - n + 1)]
        print(grams)
        return [' '.join(x) for x in grams]

    @staticmethod
    def n_grams_with_idx(words_vector, n):
        grams = [words_vector[i:i + n] for i in range(len(words_vector) - n + 1)]
        return [(' '.join([pair[0] for pair in x]), (x[0][1], x[-1][1]+len(x[-1][0]))) for x in grams]

    @staticmethod
    def n_grams_context_producer_simple(context, n_gram=2):
        context_tokens = context.strip().split(' ')
        #context_tokens = [w for w in context_tokens if w not in STOPWORDS]
        n_grams_context = []
        for i in range(n_gram):
            n_gram_content = ParsingUtils.n_grams(context_tokens, n_gram-i)
            n_grams_context.append(n_gram_content)
        return n_grams_context

    @staticmethod
    def n_grams_n_words_extractor(context, n_gram=3):
        context_tokens = context.strip().split(' ')
        context_tokens_with_index_info=[]
        word_idx=0
        for word in context_tokens:
            context_tokens_with_index_info.append((word, word_idx))
            word_idx += len(word) + 1
        #context_tokens = [w for w in context_tokens if w not in STOPWORDS]
        n_grams_context = []
        for i in range(n_gram):
            n_gram_content = ParsingUtils.n_grams_with_idx(context_tokens_with_index_info, n_gram-i)
            n_grams_context.extend(n_gram_content)
        return n_grams_context
