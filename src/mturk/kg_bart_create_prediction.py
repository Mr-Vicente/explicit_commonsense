
from utils import read_txt_2_list
import csv

if __name__ == '__main__':
    kg_bart_preds = read_txt_2_list('kg-bart_predictions_commongen.txt', 'kgbart')
    concepts_s = read_txt_2_list('concepts.txt')
    with open('kgbart/predictions.csv', 'w') as f:
        predicitons_writer = csv.writer(f)
        for concepts, kg_bart_pred in zip(concepts_s, kg_bart_preds):
            predicitons_writer.writerow([concepts,
                                        kg_bart_pred,
                                        ""])