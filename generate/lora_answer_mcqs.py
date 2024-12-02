import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, gptq_quantization, lazy_load
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


def main(
    prompt: str = "What food do llamas eat?",
    input: str = "",
    lora_path: Path = Path("out/lora/llama/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-13b-chat-hf"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 1,
    precision: Optional[str] = None,
    mcqs_dataset: Path = Path("data/MCQs/MCQs_PL_all.json"),
    results_dir: str = "results",
    with_prompt: bool = True,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    #fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    #fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    # fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    merge_lora_weights(model)
    model = fabric.setup(model)

    L.seed_everything(1234)


    # change prompt for every MCQ
    questions, answers = get_prompts(add_prompt=with_prompt, file_path=mcqs_dataset)

    quantization_string = "base"
    if quantize is not None:
        quantization_string = "quantized"
    out_dir = Path("data/" + results_dir + "/" + checkpoint_dir.name + "/" + quantization_string)
    file_path = Path(str(out_dir) + "/" + str(mcqs_dataset.name).replace(".json", "") + "/" + str(lora_path.name).replace(".pth", "") + ".json")
    # if file_path does not exist, create it
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    f = open(file_path, "w")
    f.write("[")
    for z, prompt in enumerate(questions):
        tokenizer = Tokenizer(checkpoint_dir)
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = max_returned_tokens
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        
        #fabric.print("\n" + tokenizer.decode(y))
        #fabric.print("Correct answer: " + str(answers[i]))

        # write to json file
        f.write("{")
        f.write("\"question\": \"" + prompt.replace("\n", " ") + "\",")
        f.write("\"answers\": \"" + tokenizer.decode(y).replace(prompt,"").replace("\n", " ").replace("\"", "'").replace("\\", "").replace("\t", " ") + "\",")
        # answers[i] is an array of strings, create a string with all the answers separated by a comma
        answers_string = ""
        for j, answer in enumerate(answers[z]):
            answers_string += answer
            if j < len(answers[z]) - 1:
                answers_string += ", "
        f.write("\"correct_answers\": \"" + answers_string + "\"")

        if z < len(questions) - 1:
            f.write("},")
        else:
            f.write("}")

        """ tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        ) """
        """ if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr) """
        
        print("MCQ " + str(z) + "/" + str(len(questions)) + " done", end="\r")
    f.write("]")
    f.close()



def get_prompts(add_prompt:bool=False, file_path:str="data/MCQs/MCQs_PL_all.json"):
    questions = []
    answers = []

    init_prompt = ""
    end_prompt = ""

    if add_prompt:
        init_prompt = "Answer the following question and provide the correct answer. The question is a multiple-choice question with a unique correct answer.\n"
        end_prompt = "\nAnswer the question by providing the correct alternative. Do not provide an empty answer."
        

    import json

    with open(file_path) as json_file:
        data = json.load(json_file)
        for mcq in data:
            # append answer options to question 
            options = ""
            for i, option in enumerate(mcq["options"]):
                # get alphabetic letter corresponding to i
                letter = chr(97 + i)
                options += "\n" + letter + ") " + mcq["options"][letter]
            questions.append(init_prompt + mcq["question"] + options + end_prompt + "\nAnswer: ")
            answers.append(mcq["answers"])
    
    return questions, answers

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
