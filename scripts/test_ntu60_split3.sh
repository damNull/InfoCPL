max_k=60
pretrained_path=pretrained_weights/3_best.pt
args="track=main phrase=test save_path=$pretrained_path language_path=\"data/lang_features/NTU60.npy\" language_size=768 training_strategy=\"mg_attention_multi_ms\" testing_strategy=\"mg_attention_multi_ms\" estimator_type=\"raw\" max_k=$max_k k_sep=5"

script_name=procedure_multi_desc.py
python $script_name with 'split="3"' $args