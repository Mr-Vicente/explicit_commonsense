#############################
#   Imports
#############################

# Python modules
import re
from collections import deque
import string
import os
import random

# Remote modules
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# Local modules
from utils import (read_jsonl_file_2_dict,
                   write_dict_2_json_file,
                   Data_Type)

class Data_Cleaner:
    DATA_SPLITS = ['train', 'validation', 'test']
    url_pattern = re.compile('\(_url_[0-9]_\s?\)\s?(\(.*?\))?')
    url_pattern_2 = re.compile('_url_[0-9]_\s?\s?(\(.*?\))?')
    eli5_pattern = re.compile('([\w\s\',()":-]*\s?((non-)?eli5|li5)\s?[\w\s\',()":-]*.?)')
    reddit_pattern = re.compile('([\w\s\',()":-]*\s?(\s/?r\/)\s?[\w\s\',()"-]*.?)')
    reddit_user_pattern = re.compile('([\w\s\',()":-]*\s(\/?u\/)\s?[\w\s\',()"-]*.?)')
    brackets_pattern = re.compile('([\w\s\',()\":\-]*\s?(\[(.*?)\])\s?[\w\s\',()\"-]*.?)')
    like_this_pattern = re.compile('(\[like this\])')
    not_sure_this_pattern = re.compile('([\w\s\',()":-]*not sure what you mean[\w\s\',()":-]*)')
    pontuation = string.punctuation
    split_ponctuation = '[!|?|.]'

    full_url_pattern = re.compile('(\(?([Ss]ee)?\s?((http|ftp|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])\)?)')
    site_pattern = re.compile('([\w\s\',()\":-]*\s?(site|website|webpage|link|hyperlink)\s?[\w\s\',()\":-]*.?)')
    reddit_keywords_pattern = re.compile('(reddit|subreddit|sub reddit)')

    forums_domain_language_pattern = re.compile('(thread[s]?)')
    eli5_ref_pattern = re.compile('(five-year-old|five year[s]? old)')
    eli5_ref_pattern_2 = re.compile('(like you\'re five|like you are five|like you were five)')

    def __init__(self):
        pass

    @staticmethod
    def sometimes_remove_common_pattern(answer):
        i = random.randint(0,10)
        # 7,8,9,10
        if i >= 7:
            return Data_Cleaner.remove_not_sure_pattern_from_answer(answer)
        else:
            return answer

    @staticmethod
    def remove_not_sure_pattern_from_answer(answer):
        splited_answer_on_ponctuation = re.split(Data_Cleaner.split_ponctuation, answer)
        cleaned_sentences = [fragment for fragment in splited_answer_on_ponctuation
                             if not re.match(Data_Cleaner.not_sure_this_pattern, fragment)]
        new_answer = '.'.join(cleaned_sentences)
        return new_answer

    @staticmethod
    def clean_web_related_text(answer):
        new_answer = re.sub(Data_Cleaner.full_url_pattern, '', answer)
        new_answer = re.sub(Data_Cleaner.url_pattern, '', new_answer)
        new_answer = re.sub(Data_Cleaner.url_pattern_2, '', new_answer) #the web?
        new_answer = re.sub(Data_Cleaner.site_pattern, '', new_answer)
        return new_answer

    @staticmethod
    def text_structure_cleansing(answer):
        new_answer = answer.strip()
        new_answer = re.sub("\s\s+", " ", new_answer)
        return new_answer

    @staticmethod
    def clean_forums_domain_language(answer):
        new_answer = re.sub(Data_Cleaner.forums_domain_language_pattern, 'place', answer)
        new_answer = re.sub(Data_Cleaner.eli5_ref_pattern, 'non-expert', new_answer)
        new_answer = re.sub(Data_Cleaner.eli5_ref_pattern_2, 'as a non-expert person', new_answer)
        return new_answer

    @staticmethod
    def clean_reddit_web_text(answer):
        new_answer = re.sub(Data_Cleaner.like_this_pattern, '', answer)
        new_answer = re.sub(Data_Cleaner.eli5_pattern, '', new_answer)
        new_answer = re.sub(Data_Cleaner.reddit_pattern, '', new_answer)
        new_answer = re.sub(Data_Cleaner.reddit_user_pattern, '', new_answer)
        new_answer = re.sub(Data_Cleaner.brackets_pattern, '', new_answer)
        new_answer = re.sub(Data_Cleaner.reddit_keywords_pattern, 'place', new_answer)
        new_answer = new_answer.replace('edit:', '')
        return new_answer

    def build_and_clean_dataset(self, data_type: Data_Type):
        if data_type == Data_Type.ELI5:
            self.build_and_clean_eli5_data()
        elif data_type == Data_Type.STACK_EXCHANGE:
            self.build_and_clean_stackexchange_data()
        elif data_type == Data_Type.COMMONSENSE_QA:
            self.build_and_clean_commonsense_qa_data()

    def build_and_clean_datasets(self, DATASETS):
        for dataset in DATASETS:
            print(f'Currently building {dataset} dataset')
            self.build_and_clean_dataset(dataset)
            print(f'====================================')

    def build_and_clean_eli5_data(self):
        # load data from huggingface
        data_all_splits = load_dataset('eli5')
        for data_split in Data_Cleaner.DATA_SPLITS:
            new_data = deque()
            data = data_all_splits[f'{data_split}_eli5']
            for qa_instance in tqdm(data):
                question = qa_instance['title']
                answers = qa_instance["answers"]["text"]
                for idx, answer in enumerate(answers):
                    answer = answer.lower()
                    # normally this are questions which are strongly related to images
                    if '[this]' in answer:
                        continue
                    new_answer = Data_Cleaner.clean_reddit_web_text(answer)
                    new_answer = Data_Cleaner.clean_web_related_text(new_answer)
                    #new_answer = Data_Cleaner.remove_not_sure_pattern_from_answer(new_answer)
                    new_answer = Data_Cleaner.clean_forums_domain_language(new_answer)
                    new_answer = Data_Cleaner.sometimes_remove_common_pattern(new_answer)
                    new_answer = Data_Cleaner.text_structure_cleansing(new_answer)
                    qa_instance["answers"]["text"][idx] = new_answer
                new_data_point = {
                    'question': question,
                    'answers': {
                        'text': answers,
                        'score': qa_instance["answers"]["score"]
                    }
                }
                new_data.append(new_data_point)
            new_data = list(new_data)
            random.shuffle(new_data)
            write_dict_2_json_file(new_data, f'{data_split}_eli5.json')

    def build_and_clean_stackexchange_data(self, stack_exchange_path="stackexchange"):
        stack_files = os.listdir(stack_exchange_path)
        new_data = deque()
        for stack_filename in stack_files:
            print(f'Currently processing {stack_filename} file')
            stack_filepath = f'{stack_exchange_path}/{stack_filename}'
            stack_category_content = read_jsonl_file_2_dict(stack_filepath)
            for stack_content in stack_category_content:
                question = stack_content.get('question')
                answer = stack_content.get('answer_cleaned')
                answer = answer.lower()
                new_answer = Data_Cleaner.clean_reddit_web_text(answer)
                new_answer = Data_Cleaner.clean_web_related_text(new_answer)
                new_answer = Data_Cleaner.clean_forums_domain_language(new_answer)
                new_answer = Data_Cleaner.sometimes_remove_common_pattern(new_answer)
                new_answer = Data_Cleaner.text_structure_cleansing(new_answer)
                score = stack_content.get('answer_score')
                new_data_point = {
                    'question': question,
                    'answers': {
                        'text': [new_answer],
                        'score': [score]
                    }
                }
                new_data.append(new_data_point)
        new_data = list(new_data)
        random.shuffle(new_data)
        write_dict_2_json_file(new_data, 'stackexchange_final.json')

    def build_and_clean_commonsense_qa_data(self, data_path="commonsense_qa"):
        files = os.listdir(data_path)
        new_data = deque()
        for filename in files:
            print(f'Currently processing {filename} file')
            filepath = f'{data_path}/{filename}'
            data = read_jsonl_file_2_dict(filepath)
            LABELS = ['A', 'B', 'C', 'D', 'E']
            for data_point in data:
                _qid = data_point['id']
                question = data_point['question']['stem']
                # garantee the order of labels A,B,C,D,E and obtain corresponding answers
                context = [choice['text'] for choice in data_point['question']['choices']]
                answers = np.array(
                    [choice['text'] for choice in
                     sorted(data_point['question']['choices'], key=lambda c: c['label'])])

                # the test set has no answer key so use 'A' as a dummy label
                # label = self.LABELS.index(line.get('answerKey', 'A'))
                label = answers[LABELS.index(data_point.get('answerKey', 'A'))]
                new_data_point = {
                    'question': question,
                    'context': context,
                    'answers': {
                        'text': [label],
                        'score': [1]
                    }
                }
                new_data.append(new_data_point)
        new_data = list(new_data)
        random.shuffle(new_data)
        write_dict_2_json_file(new_data, 'commonsense_qa_final.json')
