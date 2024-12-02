#!/bin/bash
models=('Llama-2-7b-chat-hf' 'Llama-2-13b-chat-hf')
quantized=('quantized' 'base')
finetuning_datasets=('book_dataset')

for model in "${models[@]}"
do
    for dataset in "${finetuning_datasets[@]}"
    do
        for q in "${quantized[@]}"
        do
            echo "${model} ${q} ${dataset}"
            if [ "$q" == "base" ]; then
                python finetune/lora_custom.py --checkpoint_dir "checkpoints/meta-llama/${model}" --out_dir "out/lora/${dataset}/f_${q}"
            else
                python finetune/lora_custom.py --checkpoint_dir "checkpoints/meta-llama/${model}" --out_dir "out/lora/${dataset}/f_${q}" --quantize bnb.nf4 --precision bf16-true
            fi
        done
    done
done
