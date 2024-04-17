CONFIG = {
    'smoothllm': {
        'pert_type': 'RandomSwapPerturbation',
        'pert_pct': 10,
        'num_copies': 10
    },
    'vicuna': {
        'model_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'tokenizer_path': '/home/e18118/fyp/smoothllm/smooth-llm/models/vicuna-7b-v1.3',
        'conversation_template': 'vicuna'
    }
}

'''
##Pert Type##
1.RandomSwapPerturbation
2.RandomPatchPerturbation
3.RandomInsertPerturbation

##Pert pct##
Initial = 10

##Num Copies##
Initial = 10
'''
