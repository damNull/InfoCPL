max_k=25
split=5
pretrained_path=pretrained_weights/${split}_best.pt
args="track=main phrase=test save_path=$pretrained_path language_path=\"data/lang_features/NTU120.npy\" language_size=768 training_strategy=\"mg_attention_multi_ms\" testing_strategy=\"mg_attention_multi_ms\" estimator_type=\"raw\" max_k=$max_k k_sep=5 dataset=ntu120"

script_name=procedure_multi_desc.py
python $script_name with 'split="5"' $args