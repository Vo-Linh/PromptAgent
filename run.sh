CUDA_VISIBLE_DEVICES=1,0 python src/main.py \
        --task_name bigbench \
        --search_algo mcts \
        --batch_size 5 \
        --depth_limit 5 \
        --train_size 70 \
        --eval_size 50 \
        --test_size 0 \
        --seed 42 \
        --train_shuffle True \
        --iteration_num 10 \
        --expand_width 3 \
        --post_instruction False \
        --pred_model nvidia/Llama3-ChatQA-1.5-8B \
        --optim_model nvidia/Llama3-ChatQA-1.5-8B \
        --log_dir logs/ \
        --data_dir datasets/penguins_in_a_table.json \
        --init_prompt "Answer questions about a table of penguins and their attributes." \
        --api_key "OPENAI-KEY"