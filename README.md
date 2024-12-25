# Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs
This repository is the code base for the paper: [Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs](TODO add link for arXiv or conference). The code is based on the [litgpt](https://github.com/Lightning-AI/litgpt) repository.

## Getting started
Clone the repository and install the requirements:
```bash
git clone https://github.com/biancaraimondi/LLaMA2_for_MCQs.git
cd LLaMA2_for_MCQs
pip install requirements.txt -r
```

## Download the models and convert the weights
As suggested in the [litgpt](https://github.com/Lightning-AI/litgpt) repository, download the models using the following command:
```bash
python scripts/download.py --repo_id <model_name>
```
where `<model_name>` is the name of the model you want to download. For example: `meta-llama/Llama-2-7b-chat-hf`.
Note that certain models require that you've been granted access to the weights on the Hugging Face Hub. In that case, you need to specify your Hugging Face access token:
```bash
python scripts/download.py --repo_id <model_name> --access_token <your_hf_token>
```

Then convert the weights to the right format:
```bash
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/<model_name>
```

## Generate the data
Use the `dataset_pipeline.sh` script if you want to create the dataset from markdown file.
Insert the path to the markdown file in the script variable `filename`.

```bash
./dataset_pipeline.sh
```

## Finetune the model
Use the `finetune.sh` script to finetune the model on the generated data.
Insert the following informations in the script variables:
- `models`: the list of models you want to finetune
- `quantized`: indicate `base` and/or `quantized` depending on the quantization you want to use
- `finetuning_datasets`: the list of datasets you want to use for finetuning

```bash
./finetune.sh
```

## Predict
Use the finetuned models to make predictions on the MCQs dataset. Use the `answer_mcqs.sh` script to make predictions.
Insert the following informations in the script variables:
- `temperatures`: the list of temperatures you want to use for prediction
- `modes`: indicate `pretrained` and/or `finetuned` depending on the version of the model you want to use
- `quantized`: indicate `base` and/or `quantized` depending on the quantization you want to use
- `f_quantized`: indicate `base` and/or `quantized` depending on the version of the finetuned model you want to use
- `finetuning_datasets`: the list of datasets you want to use
- `datasets`: the list of datasets you want to use for prediction
- `LRs`: the list of learning rates used during finetuning
- `BSs`: the list of batch sizes used during finetuning
- `MBSs`: the list of microbatch sizes used during finetuning
- `MIs`: the list of max iterations used during finetuning

```bash
./answer_mcqs.py
```

## Results analysis
Use the `results.sh` script to create the results file.
```bash
./results.sh
```
Then use the `plot.ipynb` notebook to plot the results.

## License
This repository is released under the [Apache 2.0](https://github.com/biancaraimondi/LLaMA2_for_MCQs/blob/main/LICENSE) license.

## Citation
If you use this code in your research, please cite the following paper:
```
@article{TODO citation,
  title={Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs},
  author={TODO add authors},
  journal={TODO add journal},
  year={TODO add year}
}
```
