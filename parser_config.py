import os,sys, re

# example command: 
#python bluebert/run_bluebert_ner_predict.py --data_dir=test/txt --output_dir=test/json

#--------------------- MODIFY -------------------------------------#
#  Please modifie the following parameters before parsing.

class Config():

    # Base BERT config
    max_seq_length=384 
    vocab_file="model/bluebert_config/vocab.txt"
    bert_config_file="model/bluebert_config/bert_config.json"
    pred_batch_size = 8
    learning_rate = 0.01
    
    # PICO NER config
    init_checkpoint_pico = "model/pico_model_tian_check/model.ckpt-6045"
    bluebert_pico_dir = "model/pico_model_tian_check"
    #init_checkpoint_pico = "model/pico_model/model.ckpt"
    #bluebert_pico_dir = "model/pico_model"


    # Medical Evidence Dependency config
    init_checkpoint_dependency = "model/dependency_model_tian_check/model.ckpt-20000"
    bluebert_dependency_dir = "model/dependency_model_tian_check"

    # Sentence Classification config
    init_checkpoint_sent = ""
    bluebert_sent_dir = ""

    
    # UMLS config
    use_UMLS = 0 # 0 represents not using UMLS
    QuickUMLS_git_dir = "/home/tk2624/tools/QuickUMLS-master"
    QuickUMLS_dir = "/home/tk2624/tools/QuickUMLS" # where your QuickUMLS data is intalled
    if not os.path.exists("QuickUMLS"):
        command = "ln -s "+ QuickUMLS_git_dir + " QuickUMLS"
        os.system (command)
    
