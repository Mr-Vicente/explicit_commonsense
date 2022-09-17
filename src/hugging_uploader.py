
from enum import Enum
from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
    Repository
)

class MODE(Enum):
    CREATE = 'CREATE'
    UPLOAD = 'UPLOAD'

organization = 'unlisboa'
myself = 'MrVicente'
model_name = 'bart_qa_assistant'
from transformers import BartForConditionalGeneration, BartTokenizer
PERSONAL = True
CURRENT_MODE = MODE.CREATE

def simple_operations():
    responsible = myself if PERSONAL else organization
    if CURRENT_MODE == MODE.CREATE:
        create_repo(model_name, organization=responsible, repo_type='space', space_sdk='gradio')
    elif CURRENT_MODE == MODE.UPLOAD:
        upload_file(
            "<path_to_file>/config.json",
            path_in_repo="config.json",
            repo_id=f"{responsible}/{model_name}",
        )

def repo():
    responsible = myself if PERSONAL else organization
    checkpoint_path = '/Users/mrvicente/Documents/Education/Thesis/code/models_for_aws/eli5_stackexchange_science'
    PATH = '/Users/mrvicente/Documents/Education/Thesis/code/models_for_aws/model_to_hub_2'
    repo = Repository(PATH, clone_from=f"{responsible}/{model_name}")
    repo.git_pull()
    tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.save_pretrained(repo.local_dir)
    tokenizer.save_pretrained(repo.local_dir)
    repo.git_add()
    repo.git_commit('Added model and tokenizer')
    repo.git_push()


if __name__ == '__main__':
    #repo()
    simple_operations()












