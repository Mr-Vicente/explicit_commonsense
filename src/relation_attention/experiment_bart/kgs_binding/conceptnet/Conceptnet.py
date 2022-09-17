#############################
#   Imports and Contants
#############################

# Python modules
import random
from collections import deque

# Remote modules
import conceptnet_lite
from conceptnet_lite import Label, edges_for, Language, edges_from, edges_between
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Local modules

#############################
#      CONCEPTNET_HANDLER
#############################

class Concept_Net:
    def __init__(self, database="/home/fm.vicente/thesis_datasets/Knowledge_Graphs/conceptnet.db"):
        conceptnet_lite.connect(database)
        self.st = WordNetLemmatizer()

    def build_knowledge_instance(self, entity):
        connections = edges_for(Label.get(text=entity, language='en').concepts, same_language=True)
        connections = list(connections)
        print(entity)
        one_knowledge = random.choice(connections)
        subject = one_knowledge.start.text
        predicate = one_knowledge.relation.name
        obj = one_knowledge.end.text
        print(subject, "::", obj, "|", predicate)
        # {'sub': subject, 'pred': predicate, 'obj': obj}
        return {'sub': subject, 'pred': predicate, 'obj': obj}

    def get_english_edges_for(self, entity):
        try:
            label = Label.get(text=entity, language='en')
        except Exception as _:
            return []
        return list(edges_for(label.concepts, same_language=True))

    def plural_2_singular(self, ent):
        try:
            noun = self.st.lemmatize(ent, pos='n')
            noun = self.st.lemmatize(noun, pos='v')
        except Exception as _:
            noun = ent[:-1] if ent[-1] == 's' else ent
        return noun

    def get_english_edges_between(self, entity_1, entity_2):
        try:
            #entity_1, entity_2 = self.plural_2_singular(entity_1), self.plural_2_singular(entity_2)
            label_1 = Label.get(text=entity_1, language='en')
            label_2 = Label.get(text=entity_2, language='en')
        except Exception as _:
            return []
        relations = list(edges_between(label_1.concepts, label_2.concepts))
        relations_cleaned = [r_name.relation.name for r_name in relations]
        if len(relations_cleaned) >= 2:
            for r in relations_cleaned:
                if r != 'related_to':
                    return r
            return relations_cleaned[0]
        elif relations_cleaned:
            return relations_cleaned[0]
        else:
            return None

    def simple_knowledge_prediction(self, knowledge):
        kw = list(knowledge)
        idx = random.randint(0, len(knowledge)-1) # 0-1-2
        kw[idx] = '<mask>'
        textual_knowledge_input = f'{kw[0]} {kw[1]} {kw[2]}'
        label = f'{knowledge[0]} {knowledge[1]} {knowledge[2]}'
        return f'{textual_knowledge_input},{label}\n', label

    def build_knowledge_instances(self, entity):
        knowledges = deque()
        if entity in ['base', 'post', 'set', 'check', 'light', 'lead']:
            return []
        connections = self.get_english_edges_for(entity)
        for one_knowledge in connections:
            subject = one_knowledge.start.text
            predicate = one_knowledge.relation.name
            obj = one_knowledge.end.text
            subject = subject.replace('_', ' ')
            predicate = predicate.replace('_', ' ')
            obj = obj.replace('_', ' ')
            knowledge = (subject, predicate, obj)
            knowledges.append(knowledge)
        return list(knowledges)

    def transversing_language(self, language='en', limit=1000000):
        mylanguage = Language.get(name=language)
        counter = 0
        no_reps = {}
        with open(f'bart_input/conceptnet_bart.txt', 'w') as f:
            #f.write('source,target\n')
            for l in mylanguage.labels:
                print("  Label:", l.text)
                knowledges = self.build_knowledge_instances(l.text)
                for knowledge in knowledges:
                    counter+=1
                    knowledge_text, label = self.simple_knowledge_prediction(knowledge)
                    exists = no_reps.get(label, None)
                    if exists is not None:
                        continue
                    no_reps[label] = ""
                    f.write(knowledge_text)
                    if counter==limit:
                        return

    def exists_concept(self, concept):
        try:
            _label = Label.get(text=concept, language='en')
        except Exception as _:
            return False
        return True


    def get_english_edges_from(self, entity):
        return edges_from(Label.get(text=entity, language='en').concepts, same_language=True)



if __name__ == '__main__':
    c = Concept_Net()
    knowledges = c.get_english_edges_between("apple", "worm")
    #knowledges = [x.relation.name for x in knowledges]
    print(knowledges)
    #c.transversing_language(PredictionType.RELATION)
    #print(knowledge)
    #x = Label.get(text="hsjdah", language='en')
    #print(x)
    z = c.st.lemmatize('wagging', pos='n')
    z = c.st.lemmatize(z, pos='v')
    print(z)
    w = c.plural_2_singular('wagging')
    print(w)


