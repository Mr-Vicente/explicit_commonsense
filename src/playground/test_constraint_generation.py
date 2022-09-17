from transformers import DisjunctiveConstraint
def main():
    all_constraints = []
    print('contexts:', contexts[:3])
    for context in contexts:
        constraints = []
        print('+++++++')
        concepts_from_context = relation_mapper_builder.get_concepts_from_context(context=context,
                                                                                  clear_common_wds=True)
        print('concepts_from_context:', concepts_from_context)
        useful_concepts = [relation_mapper_builder.knowledge.get_related_concepts(concept) for concept in
                           concepts_from_context]
        if not useful_concepts:
            useful_concepts = [relation_mapper_builder.kg_handler.get_related_concepts(concept) for concept in
                               concepts_from_context]
        useful_concepts = [[f'{phrase}' for phrase in concepts] for concepts in useful_concepts]  # add spaces
        # useful_concepts = [[phrase for phrase in concepts if len(phrase.split(' ')) == 1] for concepts in useful_concepts]
        # useful_concepts = list(itertools.chain.from_iterable(useful_concepts))
        # print('useful_concepts:', useful_concepts[:5])
        print('-------')
        print('useful_concepts:', useful_concepts)
        if concepts_from_context:
            for context_concept, neighbour_concepts in zip(concepts_from_context, useful_concepts):
                print('neighbour:', neighbour_concepts[:20])
                # flexible_words = self.most_similar_words(context_concept, neighbour_concepts) # limit the upperbound
                # flexible_words = [word for word in flexible_words if word not in context_concept] # remove input concepts
                flexible_words = [word for word in neighbour_concepts if
                                  word not in context_concept]  # remove input concepts
                print('flexible_words:', flexible_words)
                flexible_words_ids: List[List[int]] = tokenizer(flexible_words,
                                                                add_special_tokens=False).input_ids  # add_prefix_space=True,
                flexible_words_ids = self.remove_subsets(flexible_words_ids)
                # add_prefix_space=True
                # flexible_words_ids = [x for x in flexible_words_ids if len(x) == 1] # problem with subsets
                flexible_words_ids = flexible_words_ids[:10]
                # print('flexible_words_ids:', flexible_words_ids[:3])
                constraint = DisjunctiveConstraint(flexible_words_ids)
                constraints.append(constraint)
        all_constraints.append(constraints)

    else:
        all_constraints = None
    print('all_constraints:', all_constraints)
if __name__ == '__main__':
    main()