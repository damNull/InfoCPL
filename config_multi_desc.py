import os
import numpy as np
from sacred import Experiment

ex = Experiment("SZSL", save_git_info=False)
 
@ex.config
def my_config():
    track = "main"
    phrase = "train"
    split = '1'
    dataset = "ntu60"
    lr = 1e-5
    margin = 0.1
    weight_decay = 0
    epoch_num = 100
    batch_size = 128
    work_dir = './work_dir'
    save_name = 'output'
    weight_path = './module/gcn/model/split_'+split+".pt"
    log_path = os.path.join(work_dir, '%s/log/split_'%save_name+split+'.log')
    save_path = os.path.join(work_dir, "%s/model/"%save_name+split+"_train.pt")
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################# downstream #############################
    split_1 = [4,19,31,47,51]
    split_2 = [12,29,32,44,59]
    split_3 = [7,20,28,39,58]
    split_123 = split_1 + split_2 + split_3
    split_4 = [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]
    split_5 = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]
    split_6 = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]
    split_7 = [1, 9, 20, 34, 50]
    split_8 = [3, 14, 29, 31, 49]
    split_9 = [2, 15, 39, 41, 43]
    unseen_label = eval('split_'+split)
    visual_size = 256
    language_size = 768
    max_frame = 50
    language_path = "./data/language/"+dataset+"_embeddings.npy"
    train_list = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_data.npy"
    train_label = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_label.npy"
    test_list = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_data.npy"
    test_label = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_label.npy"
    ############################ sota compare ############################
    sota_split = "5"
    unseen_label_5 = [10,11,19,26,56]
    unseen_label_12 = [3,5,9,12,15,40,42,47,51,56,58,59]
    unseen_label_10 = [4,13,37,43,49,65,88,95,99,106]
    unseen_label_24 = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]
    sota_unseen = eval('unseen_label_'+sota_split)
    sota_train_list = "./synse_data/split_"+sota_split+"/train.npy"
    sota_train_label = "./synse_data/split_"+sota_split+"/train_label.npy"
    sota_test_list = "./synse_data/split_"+sota_split+"/test.npy"
    sota_test_label = "./synse_data/split_"+sota_split+"/test_label.npy"
    # overwrite log and param if use SOTA track
    if track == 'sota':
        log_path = os.path.join(work_dir, '%s/log/sota_split_'%save_name+sota_split+'.log')
        save_path = os.path.join(work_dir, "%s/model/_sota_"%save_name+sota_split+"_train.pt")

    training_strategy = 'avg_feat' 
    max_k = 50
    k_sep = 5
    mg_series = [1] + np.arange(k_sep, max_k, k_sep).tolist()
    # mg_series = [1] * 10 + np.arange(5, 25, 5).tolist() * 2 + [25]
    # mg_series = [1]
    mg_series = mg_series
    estimator_type = 'raw'
    # mg_series = np.arange(10, 50+10, 10).tolist()
    training_strategy_args = {'k': 40, 'num_att': 1, 'mg_att_series': mg_series, 
                              'reverse_ep': 15, 'ortho_init': '', 'gate_type': 'static',
                              'estimator_type': estimator_type} # ortho_init = '' or 'unify' or 'each

    testing_strategy = 'sum' 
    testing_strategy_args = {'k': 100, 'num_att': 3, 'mg_att_series': mg_series, 'projection': 'all', 
                             'gate_type': 'v'} # projection = 'first' or 'avg' or 'all'
    global_discriminator_arch = 'raw' # raw, fc_residual
    save_all_param = False

    # InfoNCE relelvant
    sub_loss_margin = 0.3
    sub_loss_weight = 0.5
    neg_samples = 8
    # sub positive sample relevant
    temp_mask = 5
    max_remove_joint = 20
    remove_joint = 9
# %%
