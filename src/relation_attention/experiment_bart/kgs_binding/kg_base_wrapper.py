
#############################
#   Imports
#############################

# Python modules
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

# Remote modules
from nltk.stem import WordNetLemmatizer

# Local modules

#############################
#   Constants
#############################

class KGBaseHandler(ABC):
    def __init__(self):
        super().__init__()
        self.st = WordNetLemmatizer()

    def normalize_noun(self, ent):
        try:
            noun = self.st.lemmatize(ent, pos='n')
            noun = self.st.lemmatize(noun, pos='v')
        except Exception as _:
            noun = ent[:-1] if ent[-1] == 's' else ent
        return noun

    def normalize_nouns(self, ent):
        local_ent = ent[:]
        nouns = local_ent.split(' ')
        if len(nouns) == 1:
            return ' '.join([self.normalize_noun(e) for e in nouns])
        return local_ent

    def ignore_less_relevant_connection(self, relations):
        if len(relations) >= 2:
            for r in relations:
                if r != 'related_to':
                    return r
        return relations[0]

    @abstractmethod
    def get_relation_types(self) -> List[str]:
        pass

    @abstractmethod
    def exists_relation_between(self, concept, other_concept) -> bool:
        pass

    @abstractmethod
    def relation_between(self, concept, other_concept) -> Tuple[Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def get_related_concepts(self, concept) -> Optional[List[str]]:
        pass

    @abstractmethod
    def does_concept_exist(self, concept) -> bool:
        pass

class NoKnowledge(KGBaseHandler):
    def __init__(self):
        super(NoKnowledge, self).__init__()

    def get_relation_types(self) -> List[str]:
        return []

    def exists_relation_between(self, concept, other_concept) -> bool:
        return False

    def relation_between(self, concept, other_concept) -> Tuple[Optional[str], Optional[str]]:
        return (None, None)

    def does_concept_exist(self, concept) -> bool:
        return False
