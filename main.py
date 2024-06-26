import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs

def main(args):

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Instantiate the targeted LLM
    config = model_configs.MODELS[args.target_model]
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
        # device='cpu'
    )

    # Create SmoothLLM instance
    # defense = defenses.SmoothLLM(
    #     target_model=target_model,
    #     pert_type=args.smoothllm_pert_type,
    #     pert_pct=args.smoothllm_pert_pct,
    #     num_copies=args.smoothllm_num_copies
    # )

    defense = vars(defenses)[args.defense](
        target_model=target_model
        # pert_type=args.smoothllm_pert_type,
        # pert_pct=args.smoothllm_pert_pct,
        # num_copies=args.smoothllm_num_copies
    )

    # Create attack instance, used to create prompts
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model
    )
    if(args.defense=='SmoothLLM'):


        jailbroken_results = []
        for i, prompt in tqdm(enumerate(attack.prompts)):
            output = defense(prompt)
            jb = defense.is_jailbroken(output)
            jailbroken_results.append(jb)

        # Save results to a pandas DataFrame
        summary_df = pd.DataFrame.from_dict({
            'Number of smoothing copies': [model_configs.DEFENSES['smoothllm']['num_copies']],
            'Perturbation type': [model_configs.DEFENSES['smoothllm']['pert_type']],
            'Perturbation percentage': model_configs.DEFENSES['smoothllm']['pert_pct'],
            'JB percentage': [np.mean(jailbroken_results) * 100],
            'Trial index': [args.trial]
        })
    else:
        jailbroken_results = []
        for i, prompt in tqdm(enumerate(attack.prompts)):
            output = defense(prompt)
            jailbroken_results.append(output)

        # Save results to a pandas DataFrame
        summary_df = pd.DataFrame.from_dict({
            'Number of smoothing copies': [model_configs.DEFENSES['smoothllm']['num_copies']],
            'JB percentage': [np.mean(jailbroken_results)* 100],
            'Trial index': [args.trial]
        })

    summary_df.to_pickle(os.path.join(
        args.results_dir, 'summary.pd'
    ))
    print(summary_df)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2','openOrca']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR','NoDefensePrompt','SandwitchPrompt']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    parser.add_argument(
        '--defense',
        type=str,
        default='SmoothLLM',
        choices=[
            'SmoothLLM',
            'NoPertub',
            'LlmBased'
        ]
    )

    args = parser.parse_args()
    main(args)
