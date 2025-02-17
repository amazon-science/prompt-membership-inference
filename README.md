# Has My System Prompt Been Used? Large Language Model Prompt Membership Inference

<p align="center">
    <a href="https://arxiv.org/abs/2502.09974"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
        <img src="https://img.shields.io/github/license/amazon-science/ssepy" alt="Apache-2.0">
</p>

We introduce Prompt Detective, a method for verifying if a particular system prompt was used to produce generations by a language model.
Prompt Detective works by comparing two groups of text generations based on their semantic embeddings, it performs a permutation 
test to assess the statistical significance of the difference between the mean embeddings of the two groups.

See our paper [here](https://arxiv.org/abs/2502.09974) for the technical details.

## Getting Started
To use Prompt Detective, install the package using pip. Make sure you use Python version 3.9.


```
cd prompt-membership-inference
pip install -e .
```

## Usage

Please, note that texts provided in ```group1``` and ```group2``` should be generated using similar task prompts 
and listed in the same order. For example, ```group1[i]``` and ```group2[i]``` should be generations in response to the same task prompt.
When using multiple generations per task prompts, they should be grouped together, for example if using n generations per task prompts, 
```group1[n*i: n*(i+1)]``` and ```group2[n*i: n*(i+1)]``` should be generations in response to the same task prompt.

```
from prompt_detective import PromptDetective
from datasets import Dataset

# Initialize the PromptDetective instance, specify how many task prompts and how many generations per task prompts are used.
detective = PromptDetective(model_id="bert-base-uncased", n_task_prompts=50, k_responses=1)

# Load your text generations
group1 = Dataset.from_json('sample_data/different_prompt/generations1.jsonl')['response'] # List of text generations for group 1
group2 = Dataset.from_json('sample_data/different_prompt/generations2.jsonl')['response'] # List of text generations for group 2

# Compute the p-value
p_value = detective(group1, group2, max_length=512)

# Interpret the p-value
if p_value < 0.05:
    print("The difference between the two groups is statistically significant.")
else:
    print("The difference between the two groups is not statistically significant.")

```

## Limitations

While Prompt Detective can accurately detect when two sets of generations come from different system prompts, there is
still a possibility of errors. Additionally, the Prompt Detective can conclude that two system prompts are similar only
when generations come from the same language model.

## Citation

If you use PromptDetective in your research, please cite our work:

```bibtex
@article{levin2024has,
  title={Has My System Prompt Been Used? Large Language Model Prompt Membership Inference},
  author={Levin, Roman and Cherepanova, Valeriia and Hans, Abhimanyu and Schwarzschild, Avi and Goldstein, Tom},
  journal={arXiv preprint arXiv:2502.09974},
  year={2025}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.



