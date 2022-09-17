from transformers import BartTokenizer

from custom_tokenizer.bart_custom_tokenizer_fast import BartCustomTokenizerFast
from kgs_binding.conceptnet_handler import ConceptNetHandler
from kgs_binding.relation_mapper_builder import RelationsMapperBuilder
from data.relation_utils import clean_relations

model_name = 'facebook/bart-large'

if __name__ == '__main__':
    tokenizer = BartCustomTokenizerFast.from_pretrained(model_name)
    kg_handler = ConceptNetHandler()
    relations_mapper = RelationsMapperBuilder(knowledge=kg_handler)
    relation_names = kg_handler.get_relation_types()
    tokenizer.set_known_relation_names(relation_names)
    tokenizer.set_operation_mode(there_is_difference_between_relations=True)
    #text = 'The race car is moving rather fast'
    #text = 'The north pole is where santa is from'
    #print('exp:',tokenizer([" sun", " heat"])['input_ids'])
    #text = "silhouette sun church</s>cross heat pews catholic"
    #text = "A little marten is climbing a tree"
    text = "boy garden ball house dog"
    #text = "tall sky building jumping airplane die flower funeral cry crash"
    commonsense_relation = relations_mapper.get_relations_mapping_complex(context=[text], clear_common_wds=True)
    commonsense_relation = clean_relations(commonsense_relation)[0]
    print(relations_mapper.get_kg_concepts_from_context(context=[text], clear_common_wds=True))
    print(commonsense_relation)
    source = tokenizer(text, padding='max_length',
        truncation='longest_first', max_length=7,
        return_tensors="pt", return_offsets_mapping=True,
        input_commonsense_relations=commonsense_relation,
    )
    print('source:', source)

    x = "the cat was fleeing from the dog, after passing through it's dog house after getting scared from an airplane that passed by"
    x = "an air plane is flying over heavy clouds to transport people over an ocean"
    x = "on the air, heavy clouds shadow a plane, which is flying to transporting people over an ocean"
    y = relations_mapper.get_concepts_from_context(x, clear_common_wds=True)
    z = relations_mapper.get_kg_concepts_from_context([x], clear_common_wds=True)
    print(relations_mapper.get_relations_mapping_complex(context=[x], clear_common_wds=True))
    print(y)
    print("vs")
    print(z)
    simp_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    print(simp_tokenizer.encode_plus(("hello", "friend")))

    r = simp_tokenizer(["i want to have friends", "what about me", "cute"], return_tensors='pt', padding=True)
    print(r.get('attention_mask'))
    print(r.get('attention_mask')[0])
    u = simp_tokenizer(["i want to have friends", "what about me", "cute"], add_special_tokens=False, add_prefix_space=True).input_ids
    print(simp_tokenizer.batch_decode(u))