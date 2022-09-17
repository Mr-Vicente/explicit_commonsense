from utils import read_json_file_2_dict

if __name__ == '__main__':
    _dir = '/Users/mrvicente/Documents/Education/Thesis/code/f_papers/explicit_commonsense/src/relation_attention/experiment_bart/kgs_binding/conceptnet'
    filename = 'conceptnet_english_noun_2_noun_relations.json'
    d = read_json_file_2_dict(filename, _dir)
    rels = list(set([r for v in d.values() for r in v if r]))
    print(rels)
    print(len(rels))