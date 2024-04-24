MODELS = {
    'llama2': {
        'model_path': '/home/e18118/llm_attacks_and_defences/llm-attacks-and-defences/models/Llama-2-7b-chat',
        'tokenizer_path': '/home/e18118/llm_attacks_and_defences/llm-attacks-and-defences/models/Llama-2-7b-chat',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'tokenizer_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'conversation_template': 'vicuna'
    },
    'openOrca': {
        'model_path': '/home/e18118/llm_attacks_and_defences/llm-attacks-and-defences/models/Mistral-7B-OpenOrca',
        'tokenizer_path': '/home/e18118/llm_attacks_and_defences/llm-attacks-and-defences/models/Mistral-7B-OpenOrca',
        'conversation_template': 'mistral-7b-openorca'
    }
}

'''
---DEFENSES---

##attack##
1.NoDefensePrompt
2.GCG
3.PAIR

##defence##
1.SmoothLLM
2.NoPertub

##Pert Type##
1.RandomSwapPerturbation
2.RandomPatchPerturbation
3.RandomInsertPerturbation

##Pert pct##
Initial = 10

##Num Copies##
Initial = 10
'''
DEFENSES = {
    'smoothllm': {
        'pert_type': 'RandomSwapPerturbation',
        'pert_pct': 10,
        'num_copies': 5
    },
    'vicuna': {
        'model_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'tokenizer_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'conversation_template': 'vicuna'
    }
}

