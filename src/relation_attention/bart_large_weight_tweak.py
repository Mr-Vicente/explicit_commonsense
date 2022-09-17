
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def load_huggingface_model_and_store(model_name='facebook/bart-large'):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    output_dir = '/home/fm.vicente/model_weights/commonsense/'
    model.save_pretrained(
        output_dir,
        is_main_process=True,
        state_dict=model.state_dict(),
        save_function=xm.save,
    )
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    # please download the model in your local directory and manually change the weight
    state_dict = torch.load("bart-large/pytorch_model.bin")
    state_dict["model.shared.weight"][0] = state_dict["model.shared.weight"][0] + torch.randn(1024)
    torch.save(state_dict, open("bart-large/pytorch_model.bin", "wb"))