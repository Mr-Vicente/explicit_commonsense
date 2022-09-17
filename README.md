<!-- PROJECT SHIELDS -->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Mr-Vicente/NS-BART">
    <img src="repo_images/c_bart.png" alt="Logo" width="125" height="125">
  </a>

  <h1 align="center">Commonsense BART</h1>

  <p align="center">
    Commonsense enhenced language model !
    <br />
    <a href="https://github.com/Mr-Vicente/explicit_commonsense"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/spaces/MrVicente/RA-BART">View Demo</a>
    ·
    <a href="https://github.com/Mr-Vicente/explicit_commonsense/issues">Report Bug</a>
    ·
    <a href="https://github.com/Mr-Vicente/explicit_commonsense/issues">Request Feature</a>
  </p>
</p>

## Description

Commonsense BART is a Python library which provides a relation-aware BART Model enriched with Commonsense Knowledge.

## Installation

You can clone with git to install this Commonsense BART library.
```bash
git clone https://github.com/Mr-Vicente/explicit_commonsense.git
```

## Usage

```python
from inference import RelationsInference
from utils import KGType, Model_Type

qa_bart = RelationsInference(
    model_path='MrVicente/commonsense_bart_absqa',
    kg_type=KGType.CONCEPTNET,
    model_type=Model_Type.RELATIONS,
    max_length=128
)
# ask something to the model
question = "What is the meaning of life?"
response, _, _ = qa_bart.generate_based_on_context(question)
answer = response[0]
print(answer)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)