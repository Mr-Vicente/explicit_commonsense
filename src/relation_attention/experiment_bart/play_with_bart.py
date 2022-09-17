
from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from custom_bart.bart_for_conditional_generation import BartCustomForConditionalGeneration
from custom_bart.config import BartCustomConfig


if __name__ == '__main__':
    model_name = 'facebook/bart-large'
    tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
    relation_names = ['derived_from', 'dbpedia/genre', 'antonym', 'desires', 'related_to', 'part_of',
                    'used_for', 'causes_desire', 'capable_of', 'has_context', 'entails', 'motivated_by_goal',
                    'causes', 'distinct_from', 'made_of', 'synonym', 'at_location', 'manner_of',
                    'etymologically_related_to', 'has_subevent', 'has_prerequisite', 'not_desires',
                    'has_property', 'similar_to', 'is_a', 'has_a']
    tokenizer.set_known_relation_names(relation_names)
    question_sample = 'why is the dog chasing the cat?'
    max_length = 128
    relation = {
        (11, 13): {
            (27, 29): "antonym"
        },
        (27, 29): {
            (11, 13): "antonym"
        }
    },
    source = tokenizer(question_sample, padding='max_length',
                            truncation='longest_first', max_length=max_length,
                            input_commonsense_relations=relation,
                            return_tensors="pt", return_offsets_mapping=True)
    input_commonsense_relations = source.get('input_commonsense_relations', None)
    print("------Playground!!!!!!------")
    print(input_commonsense_relations)
    print(input_commonsense_relations.sum())