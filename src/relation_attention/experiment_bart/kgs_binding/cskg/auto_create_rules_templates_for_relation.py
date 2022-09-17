#############################
#   Imports
#############################

# Python modules
from tqdm import tqdm
# Remote modules

# Local modules

#############################
#   Constants
#############################

BASE_PATH = '/Users/mrvicente'

#############################
#   Stuff
#############################

def fecth_predicates_from_kb(kb_path=f'{BASE_PATH}/Documents/Education/Thesis/code/cskg_kb/cskg_kb_2.txt'):
    with open(kb_path, 'r') as f:
        predicates = set()
        for r in f:
            predicate_name = r.split('(')[0]
            predicates.add(predicate_name)
        return predicates

def write_transitive_rule(f, idx):
    p1_idx, p2_idx, p3_idx = idx, idx + 1, idx + 2
    f.write(f'TEMPLATE_SYMBOL_{p1_idx}(X,Z) :- TEMPLATE_SYMBOL_{p2_idx}(X,Y), TEMPLATE_SYMBOL_{p3_idx}(Y,Z).\n')
    new_idx = idx + 3
    return new_idx

def write_inversive_rule(f, idx):
    p1_idx, p2_idx = idx, idx + 1
    f.write(f'TEMPLATE_SYMBOL_{p1_idx}(X,Y) :- TEMPLATE_SYMBOL_{p2_idx}(Y,X).\n')
    new_idx = idx + 2
    return new_idx

def write_self_rule(f, predicate_name, idx):
    f.write(f'{predicate_name}(X,Y) :- TEMPLATE_SYMBOL_{idx}(X,Y).\n')
    new_idx = idx + 1
    return new_idx

def write_rules_templates(predicates,
                          rules_path=f'{BASE_PATH}/Documents/Education/Thesis/code/NS-BART/src/ns_bart/reasoner/rules/cskg_auto.txt'):
    pred_idx = 0
    with open(rules_path, 'w') as f:
        for idx, predicate_name in enumerate(predicates):
            pred_idx = write_transitive_rule(f, pred_idx)
            pred_idx = write_transitive_rule(f, pred_idx)
            pred_idx = write_inversive_rule(f, pred_idx)
            pred_idx = write_inversive_rule(f, pred_idx)
            pred_idx = write_self_rule(f, predicate_name, pred_idx)
            pred_idx = write_self_rule(f, predicate_name, pred_idx)


if __name__ == '__main__':
    predicates = list(fecth_predicates_from_kb())[1:]
    print(predicates)
    print(len(predicates))
    write_rules_templates(predicates)