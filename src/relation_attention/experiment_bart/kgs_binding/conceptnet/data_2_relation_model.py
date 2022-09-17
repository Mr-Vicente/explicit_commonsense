import random

from Conceptnet import Concept_Net
from src.utils import   (read_json_file_2_dict,
                        read_simple_text_file_2_vec,
                        write_dict_2_json_file,
                        Data_Type)
from typing import List, Dict

import re
import string
from collections import defaultdict
from tqdm import tqdm
from collections import deque

STOPWORDS = read_simple_text_file_2_vec('english_stopwords.txt')

def remove_pontuation(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.translate(str.maketrans('', '', string.punctuation))

def load_data(data_path='commongen_qa_final.json'):
    data = read_json_file_2_dict(data_path)
    return data

def fetch_questions_from_data(data: List[Dict], datatype:Data_Type = Data_Type.COMMONGEN_QA):
    if datatype == Data_Type.COMMONGEN_QA:
        questions = [data_unit.get('title').lower() for data_unit in data]
    elif datatype == Data_Type.ELI5:
        questions = [data_unit.get('question').lower() for data_unit in data]
    else:
        questions = []
    return questions

def two_entities_relation(conceptnet, entity_1, entity_2):
    left_2_right = conceptnet.get_english_edges_between(entity_1, entity_2)
    right_2_left = conceptnet.get_english_edges_between(entity_2, entity_1)
    return left_2_right, right_2_left

def get_word_range_mapping(context, word_token):
    word_token_start = context.index(word_token)
    word_token_end = word_token_start + len(word_token) - 1  # inclusive end
    return word_token_start, word_token_end

def get_relations_mapping(conceptnet, questions):
    context = deque()
    for q_id, question in tqdm(enumerate(questions)):
        simple_question = remove_pontuation(question)
        word_tokens = simple_question.strip().split(' ')
        word_tokens = [w for w in word_tokens if w not in STOPWORDS]
        print(f'question: {question}')
        print(f'word_tokens: ', word_tokens)
        relation_context_between_words = defaultdict(dict)
        known_tokens = set()
        for token_i, first_word_token in enumerate(word_tokens[:-1]):
            known_tokens.add(first_word_token)
            first_word_range = get_word_range_mapping(question, first_word_token)
            first_word_range_str = str(first_word_range)
            for second_word_token in [w for w in word_tokens[token_i+1:] if w not in known_tokens]:
                second_word_range = get_word_range_mapping(question, second_word_token)
                second_word_range_str = str(second_word_range)
                left_2_right, right_2_left = two_entities_relation(conceptnet, first_word_token, second_word_token)
                if left_2_right:
                    relation_context_between_words[first_word_range_str][second_word_range_str] = left_2_right
                #else:
                #    if not relation_context_between_words.get(first_word_range_str, {}).get(second_word_range_str, ''):
                #        relation_context_between_words[first_word_range_str][second_word_range_str] = 'neutral'
                if right_2_left:
                    relation_context_between_words[second_word_range_str][first_word_range_str] = right_2_left
                #else:
                #    if not relation_context_between_words.get(second_word_range_str, {}).get(first_word_range_str, ''):
                #        relation_context_between_words[second_word_range_str][first_word_range_str] = 'neutral'
        context.append(dict(relation_context_between_words))
    return list(context)


if __name__ == '__main__':
    conceptnet = Concept_Net()
    data = load_data()
    questions = fetch_questions_from_data(data)
    rel_qs = get_relations_mapping(conceptnet, questions)
    print(rel_qs)
    write_dict_2_json_file(rel_qs, 'eli5_relation_data.json')