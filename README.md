# SmoothLLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official source code for "[SmoothLLM: Defending LLMs Against Jailbreaking Attacks](https://arxiv.org/abs/2310.03684)" by [Alex Robey](https://arobey1.github.io/), [Eric Wong](https://riceric22.github.io/), [Hamed Hassani](https://www.seas.upenn.edu/~hassani/), and [George J. Pappas](https://www.georgejpappas.org/).  To learn more about our work, see [our blog post](https://debugml.github.io/smooth-llm/).

<!-- ![Overview of SmoothLLM results.](assets/overview.png) -->
![Introduction to SmoothLLM](assets/introduction.gif)

## Installation

**Step 1:** Create an empty virtual environment.

```bash
conda create -n smooth-llm python=3.10
conda activate smooth-llm
```

**Step 2:** Install the source code for "[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)."

```bash
git clone https://github.com/llm-attacks/llm-attacks.git
cd llm-attacks
pip install -e .
```

**Step 3:** Download the weights for [Vicuna](https://huggingface.co/lmsys/vicuna-13b-v1.5) and/or [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) from HuggingFace.  

**Step 4:** Change the paths to the model and tokenizer in `configs/vicuna.py` and/or `configs/llama2.py` depending on which set(s) of weights you downloaded in Step 3.  E.g., for vicuna, change the following paths in `configs/vicuna.py`:

```bash
config.model_paths = ["/path/to/vicuna-13b-v1.5"]
config.tokenizer_paths = ["/path/to/vicuna-13b-v1.5"]
```

## Experiments

We provide ten adversarial suffix generated by running GCG for Vicuna and Llama2 in the `data/` directory.  You can run SmoothLLM by running:

```bash
python main.py \
    --results_dir ./results \
    --target_model vicuna \
    --attack GCG \
    --attack_logfile data/GCG/vicuna_behaviors.json \
    --smoothllm_pert_type RandomSwapPerturbation \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 10
```

You can also change SmoothLLM's hyperparameters---the number of copies, the perturbation percentage, and the perturbation function---by changing the named arguments.  At present, we support three kinds of perturbations: swaps, patches, and insertions.  For more details, see Algorithm 2 in [our paper](https://arxiv.org/abs/2310.03684).  To use these functions, you can replace the `--perturbation_type` value with `RandomSwapPerturbation`, `RandomPatchPerturbation`, or `RandomInsertPerturbation`.

## Reproducibility
The following codebases have reimplemented our results:
* https://gist.github.com/deadbits/4ab3f807441d72a2cf3105d0aea9de48

## Citation
If you find this codebase useful in your research, please consider citing:

```bibtex
@article{robey2023smoothllm,
  title={SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks},
  author={Robey, Alexander and Wong, Eric and Hassani, Hamed and Pappas, George J},
  journal={arXiv preprint arXiv:2310.03684},
  year={2023}
}
```

## License
`smooth-llm` is licensed under the terms of the MIT license. See LICENSE for more details.
