#############################
#   Imports
#############################

# Python modules

# Remote modules
#import numpy as np
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.spice.spice import *
import pycocoevalcap.spice as spice_dir
import inspect

# Local modules

#############################
#   Constants
#############################

#############################
#   Stuff
#############################

"""
def score_individual(ref, pred):
    return 0

def conceptualy_close(predictions, references):
    scores = [
        score_indivudal(ref, pred)
        for ref, pred in zip(references, predictions)
    ]
    return {"cc": np.mean(scores)}
"""
def compute_bleu_score(bleu_metric, preds, refs, max_order=3, smooth=True):
    try:
        bleu_result = bleu_metric.compute(predictions=preds, references=refs, max_order=max_order, smooth=smooth)
    #except ZeroDivisionError as _:
    except Exception as _:
        bleu_result = {'bleu': 0.0}
    return bleu_result


def calc_accuracy(accuracy_metric, predictions, references):
    tmp = []
    for pred, ref in zip(predictions, references):
        pred_diff = len(pred) - len(ref)
        if pred_diff > 0: # pred biggest
            for i in range(pred_diff):
                ref.append(1)
        elif abs(pred_diff) > 0: # ref biggest
            for i in range(abs(pred_diff)):
                pred.append(1)
        acc = accuracy_metric.compute(predictions=pred, references=ref)
        tmp.append(acc.get("accuracy", -1) * 100)
    mean_acc = np.mean(tmp)
    return {"accuracy": mean_acc}

class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, predictions, gold_references):
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for hypo, ref in zip(predictions, gold_references):
            cider_scorer += (hypo, [ref])
        (score, scores) = cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = "spice-1.0.jar"
print('SPICE_JAR:', SPICE_JAR)
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'

class Spice:
    """
    Main Class to compute the SPICE metric
    """
    def __init__(self):
        get_stanford_models()

    def float_convert(self, obj):
        try:
            return float(obj)
        except:
            return np.nan

    def compute_score(self, predictions, gold_references):
        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id,(hypo, ref) in enumerate(zip(predictions, gold_references)):
            ref = [ref]

            input_data.append({
                "image_id": id,
                "test": hypo,
                "refs": ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir,
                                              mode='w+')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
                     '-cache', cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent'
                     ]
        #subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        d_name = f"{inspect.getfile(spice_dir).replace('/__init__.py','')}"
        subprocess.check_call(spice_cmd, cwd=d_name)

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        spice_scores = []
        for item in results:
            spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        return average_score

    def method(self):
        return "SPICE"

"""
if __name__ == '__main__':
    # test cider and spice
    #preds = ['the sky is blue', 'the cat is flying', 'the sky is shiny']
    #gold_refs = ['the sky is red', 'the dog is swimming', 'the sky is shining']

    preds = ['the sky is red', 'the cat is flying', 'the sky is shiny']
    gold_refs = ['the sky is red', 'the cat is wtf is going on', 'the sky is shiny']

    print("======= CIDEr ========")
    c = Cider()
    c_score = c.compute_score(preds, gold_refs)[1]
    print('CIDEr score:', c_score)
    print("======= ===== ========")

    #print("======= Spice ========")
    #s = Spice()
    #s_score = s.compute_score(preds, gold_refs)
    #print('Spice score:', s_score)
    #print("======= ===== ========")
"""
