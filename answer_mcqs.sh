#!/bin/bash
temperatures=(0)
modes=('pretrained' 'finetuned')
models=('Llama-2-7b-chat-hf' 'Llama-2-13b-chat-hf' 'Llama-2-70b-chat-hf')
quantized=('base' 'quantized')
f_quantized=('base' 'quantized')
finetuning_datasets=('book_dataset')
datasets=('data/MCQs/MCQs_PL_all')
LRs=('0.0001' '0.001')
BSs=('16' '32' '64' '128')
MBSs=('2')
MIs=('2000' '1500' '1000' '500' '100')

for mode in "${modes[@]}"
do
    for model in "${models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for temperature in "${temperatures[@]}"
            do
                for q in "${quantized[@]}"
                do
                    if [ $mode == "pretrained" ];
                    then
                        echo "${model} ${mode} ${dataset} ${q} temperature${temperature}"
                        file_path="lit-gpt/data/results/${mode}/temperature${temperature}/${model}/${q}/${dataset#data/MCQs/}.json"
                        if [ -f $file_path ]; then
                            echo "File ${file_path} yet exists."
                        else
                            if [ $q == "base" ];
                            then
                                python generate/answer_mcqs.py --checkpoint_dir "checkpoints/meta-llama/${model}" --mcqs_dataset "${dataset}.json" --temperature "${temperature}" --results_dir "results/${mode}/temperature${temperature}"
                            else
                                python generate/answer_mcqs.py --checkpoint_dir "checkpoints/meta-llama/${model}" --mcqs_dataset "${dataset}.json" --temperature "${temperature}" --results_dir "results/${mode}/temperature${temperature}" --quantize bnb.nf4 --precision bf16-true
                            fi
                        fi
                    else
                        for f_dataset in "${finetuning_datasets[@]}"
                        do
                            for lr in "${LRs[@]}"
                            do
                                for bs in "${BSs[@]}"
                                do
                                    for mbs in "${MBSs[@]}"
                                    do
                                        for mi in "${MIs[@]}"
                                        do
                                            for f_q in "${f_quantized[@]}"
                                            do
                                                echo "${model} ${mode} ${dataset} ${f_dataset}_${f_q} ${q} temperature${temperature} lr${lr} bs${bs} mbs${mbs} mi${mi}"
                                                lora_path="out/lora/${f_dataset}/f_${f_q}/${model}/lit_model_lora_finetuned-lr${lr}-bs${bs}-mbs${mbs}-mi${mi}.pth"
                                                if [ -f $lora_path ]; then
                                                    file_path="lit-gpt/data/results/${mode}/temperature${temperature}/${f_dataset}/f_${f_q}/${model}/${q}/${dataset#data/MCQs/}/lit_model_lora_finetuned-lr${lr}-bs${bs}-mbs${mbs}-mi${mi}.json"
                                                    if [ -f $file_path ]; then
                                                        echo "File ${file_path} yet exists."
                                                    else
                                                        echo "File ${file_path} does not exist."
                                                        if [ $q == "base" ];
                                                        then
                                                            python generate/lora_answer_mcqs.py --checkpoint_dir "checkpoints/meta-llama/${model}" --mcqs_dataset "${dataset}.json" --lora_path "out/lora/${f_dataset}/f_${f_q}/${model}/lit_model_lora_finetuned-lr${lr}-bs${bs}-mbs${mbs}-mi${mi}.pth" --temperature "${temperature}" --results_dir "results/${mode}/temperature${temperature}/${f_dataset}/f_${f_q}"
                                                        else
                                                            python generate/lora_answer_mcqs.py --checkpoint_dir "checkpoints/meta-llama/${model}" --mcqs_dataset "${dataset}.json" --lora_path "out/lora/${f_dataset}/f_${f_q}/${model}/lit_model_lora_finetuned-lr${lr}-bs${bs}-mbs${mbs}-mi${mi}.pth" --temperature "${temperature}" --results_dir "results/${mode}/temperature${temperature}/${f_dataset}/f_${f_q}" --quantize bnb.nf4 --precision bf16-true
                                                        fi
                                                    fi
                                                else
                                                    echo "File ${lora_path} does not exist."
                                                fi
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    fi
                done
            done
        done
    done
done
