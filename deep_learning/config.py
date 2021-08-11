class AlbertCfg:
    # data
    host = 'host'
    port = 1000
    data_path = 'data_path'
    username = 'username'
    password = 'password'
    query = {
        "_source": ["source_name"]
    }

    # model
    model_param = {
        'vocab_size': 30000,
        'embedding_size': 128,
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_hidden_groups': 1,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'inner_group_num': 1,
        'hidden_act': 'gelu_new',
        'hidden_dropout_prob': 0,
        'attention_probs_dropout_prob': 0,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'initializer_range': 0.01,
        'layer_norm_eps': 1e-10,
        'classifier_dropout_prob': 0.1,
        'position_embedding_type': 'absoulte',
        'bos_token_id': 1,
        'eos_token_id': 2
    }

    # other
    batch_size = 100
    learning_rate = 0.00001
    max_gram = 3
    mask_alpha = 4
    mask_beta = 1
    mask_prob = 0.15
    max_pred = 77
    min_sentence_length = 8
    max_position_embeddings = 512

    epochs = 5
    log_period = 1000

    save_path = './deep_learning/saved_model/albert'


class W2VCfg:
    host = 'host'
    port = 1000
    username = 'username'
    password = 'password'
    data_path = '/remote/data_path'
    processor_path = './w2v2_processor'

    model_param = {
        'dropout': 0.1,
        'vocab_size': 1862
    }

    # other
    batch_size = 1
    learning_rate = 0.00001

    epochs = 10
    log_period = 100

    save_path = './deep_learning/saved_model/w2v2_pretrained'
