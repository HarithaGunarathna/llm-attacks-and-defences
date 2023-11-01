from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters 
    config.target_weight=1.0
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = 'results_testing/individual_vicuna7b'

    # tokenizers
    config.tokenizer_paths=['/shared_data0/arobey1/vicuna-13b-v1.5']
    config.tokenizer_kwargs=[{"use_fast": False}]
    
    config.model_paths=['/shared_data0/arobey1/vicuna-13b-v1.5']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": True}]
    config.conversation_templates=['vicuna']
    config.devices=['cuda:0']

    config.perturbation_type = ''
    config.perturbation_percentage = 0
    config.num_smoothing_copies = 1
    config.smoothing_results_dir = 'results'
    config.logfile = ''
    config.attack = 'GCG'

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'gcg'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    # evaluation parameters
    config.perturbation_type = 'char'
    config.perturbation_pct = 5
    config.n_patches = 1
    config.w_patches = 3

    config.gbda_deterministic = True

    return config
