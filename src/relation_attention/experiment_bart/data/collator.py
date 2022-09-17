from transformers.data.data_collator import DataCollatorForSeq2Seq

class BartDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model, max_length=256):
        super(BartDataCollator, self).__init__(tokenizer=tokenizer, model=model, max_length=max_length)

    def __call__(self, features, return_tensors=None):
        results = super(BartDataCollator, self).__call__(features, return_tensors)
        return results