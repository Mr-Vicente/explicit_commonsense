#############################
#   Imports
#############################

# Python modules
from collections import deque
import json

# Remote modules
from bs4 import BeautifulSoup
from tqdm import tqdm

# Local modules
from utils import write_dict_2_json_file

#############################
#   Stuff
#############################

def process_nq_data(filename, store_dir='.', limit=-1):
    examples = deque()
    with open(f'{store_dir}/{filename}', 'r', encoding='utf-8') as file:
        for idx, line in tqdm(enumerate(file)):
            if idx == limit:
                break
            nq_example = json.loads(line)
            new_example = simplify_example(idx, nq_example)
            if new_example:
                examples.append(new_example)
        list_examples = list(examples)
        write_dict_2_json_file(list_examples, 'nq_parsed.jsonl','/Users/mrvicente/Downloads/')

def simplify_example(idx, nq_example):
    document_text = nq_example.get('document_text')
    question = nq_example.get('question_text')
    annotations = nq_example.get('annotations', [])[0]
    short_answer = annotations.get('short_answers', [])
    long_answer = annotations.get('long_answer', {})
    start_token = long_answer.get('start_token', -1)
    end_token = long_answer.get('end_token', -1)
    answer_text = " ".join(document_text.split(" ")[start_token:end_token])
    html_2_str = BeautifulSoup(answer_text, features="html.parser")
    html_2_str = html_2_str.get_text()
    if short_answer:
        short_answer = short_answer[0]
        short_s = short_answer.get('start_token', -1)
        short_e = short_answer.get('end_token', -1)
        x = " ".join(document_text.split(" ")[short_s:short_e])
        #print('short:',  x)
    if html_2_str:
        new_example = {
            'q_id': idx,
            'title': question,
            'answers': {
                'a_id': [idx],
                'text': [html_2_str],
                'scores': [1]
            }
        }
    else:
        new_example = None
    return new_example

if __name__ == '__main__':
    nq_data_filename = 'v1.0-simplified_simplified-nq-train.jsonl'
    nq_data_directory_path = '/Users/mrvicente/Downloads/'
    limitation = -1 # -1 -> no limit
    nq_examples = process_nq_data(nq_data_filename, nq_data_directory_path, limitation)


