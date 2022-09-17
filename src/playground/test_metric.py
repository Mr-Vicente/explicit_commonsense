
from datasets import load_metric
import numpy as np
from evaluate import load

def test_bleu():
    blue_metric = load('bleu')
    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
    predictions = ["the sky is red", "general kenobi"]
    references = [["the sky is blue"], ["general kenobi"]]
    predictions = ["", ""]
    references = [["the sky is blue"], ["general kenobi"]]
    try:
        bleu_result = blue_metric.compute(predictions=predictions, references=references, max_order=3)
    except ZeroDivisionError as _:
        bleu_result = {'bleu': 0.0}
    #results = blue_result['score']
    print(bleu_result)

def test_rouge():
    rouge_metric = load('rouge')
    predictions = ["hello there", "general kenobi"]
    references = ["hello", "general kenobi"]
    results = rouge_metric.compute(predictions=predictions,references=references,use_aggregator=False)
    results = results['rougeL']
    results = np.mean([s.fmeasure for s in results])
    print(results)

def test_meteor():
    metric = load('meteor')
    predictions = ["hello there", "general kenobi"]
    references = ["hello", "general kenobi"]
    #references = [["hello"], ["general kenobi"]]
    results = metric.compute(predictions=predictions,references=references)
    print(results)

def test_perplexity():
    perplexity = load("perplexity", module_type="metric")
    input_texts = ["hello there, general kenobi"]
    results = perplexity.compute(input_texts=input_texts, model_id='facebook/bart-large')
    #torch.exp(loss)
    print(results)

def test_eval_meteor():
    metric = load('meteor')
    predictions = ["dog"]
    references = ["domestic dog"]
    #references = [["hello"], ["general kenobi"]]
    results = metric.compute(predictions=predictions,references=references)
    print(results)

if __name__ == '__main__':
    test_eval_meteor()
