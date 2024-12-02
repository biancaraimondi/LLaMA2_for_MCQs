import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    tokens = [prompt]
    input_pos = torch.tensor([T], device=device)
    token = next_token(
        model, torch.arange(0, T, device=device), prompt.view(1, -1), temperature=temperature, top_k=top_k
    ).clone()
    tokens.append(token)
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k).clone()
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    return torch.cat(tokens)


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 1,
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
    add_prompt: bool = True,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    fabric = None
    model = None
    if add_prompt:
        if checkpoint_dir == Path("checkpoints/meta-llama/Llama-2-70b-chat-hf"):
            quantize = "bnb.nf4"
            precision = "bf16-true"

        precision = precision or get_default_supported_precision(training=False)

        plugins = None
        if quantize is not None and quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

        fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

        check_valid_checkpoint_dir(checkpoint_dir)

        config = Config.from_json(checkpoint_dir / "lit_config.json")

        if quantize == "gptq.int4":
            model_file = "lit_model_gptq.4bit.pth"
            if not (checkpoint_dir / model_file).is_file():
                raise ValueError("Please run `python quantize/gptq.py` first")
        else:
            model_file = "lit_model.pth"
        checkpoint_path = checkpoint_dir / model_file



        #fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
        t0 = time.perf_counter()
        with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
            model = GPT(config)
        #fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = 1000 #max_returned_tokens
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        if compile:
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._inductor.config.triton.unique_kernel_names = True
            torch._inductor.config.coordinate_descent_tuning = True
            global next_token
            next_token = torch.compile(next_token, mode="reduce-overhead")

        model = fabric.setup_module(model)

        t0 = time.perf_counter()
        load_checkpoint(fabric, model, checkpoint_path)
        #fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

        L.seed_everything(1234)

    modes = ['pretrained', 'finetuned']
    models = ['Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Llama-2-70b-chat-hf']
    temperatures = ["0"]
    quantized = ['base', 'quantized']
    f_quantized = ['f_base', 'f_quantized']
    datasets = ['MCQs_PL_all']
    finetuning_datasets=['book_3chapters_dataset', 'book_dataset', 'book_1chapter_dataset']
    LRs = ['0.001', '0.0001']
    BSs = ['16', '32', '64', '128']
    MBSs = ['2']
    MIs = ['100', '500', '1000', '1500', '2000']

    for mode in modes:
        for m in models:
            for q in quantized:
                for ds in datasets:
                    for t in temperatures:
                        print(f'{m} {mode} {q}')
                        if mode == 'pretrained':
                            results_path = "data/results/" + mode + "/temperature" + t + "/" + m + "/" + q + "/" + ds + ".json"
                            verify(add_prompt, results_path, checkpoint_dir, fabric, max_new_tokens, num_samples, temperature, top_k, model)
                        else:
                            for f_dataset in finetuning_datasets:
                                for f_q in f_quantized:
                                    for lr in LRs:
                                        for bs in BSs:
                                            for mbs in MBSs:
                                                for mi in MIs:
                                                    results_path = "data/results/" + mode + "/temperature" + t + "/" + f_dataset + "/" + f_q + "/" + m + "/" + q + "/" + ds + "/lit_model_lora_finetuned-lr" + lr + "-bs" + bs + "-mbs" + mbs + "-mi" + mi + ".json"
                                                    verify(add_prompt, results_path, checkpoint_dir, fabric, max_new_tokens, num_samples, temperature, top_k, model)

def verify(add_prompt, results_path, checkpoint_dir, fabric, max_new_tokens, num_samples, temperature, top_k, model):
    # change prompt for every MCQ
    print("\nRESULTS PATH: " + str(results_path))
    # if results_path does not exist, abort
    if not Path(results_path).exists():
        print("Results path does not exist")
        return
    prompts, functional_programming, algorithms, fundations, abstract_machines, memory_management, names_and_the_environment, describing_a_programming_language, object_oriented_paradigm, control_structure, structuring_data, programming_languages, num_functional_programming, num_algorithms, num_fundations, num_abstract_machines, num_memory_management, num_names_and_the_environment, num_describing_a_programming_language, num_object_oriented_paradigm, num_control_structure, num_structuring_data, num_programming_languages = get_verification_prompts(add_prompt=add_prompt, file_path=results_path)

    file_path = Path(str(results_path).replace("results", "verified_results"))
    # if file_path does not exist, create it
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    f = open(file_path, "w")
    f.write("[")
    for z, prompt in enumerate(prompts):
        if not add_prompt:
            f.write("{")
            f.write("\"verified_mcq\": \"" + prompt + "\",")
            f.write("\"evaluation\": \"" + prompt + "\",")
            f.write("\"evaluation_score\": \"" + prompt + "\"")
        else:
            tokenizer = Tokenizer(checkpoint_dir)
            encoded = tokenizer.encode(prompt, device=fabric.device)
            prompt_length = encoded.size(0)
            max_returned_tokens = prompt_length + max_new_tokens
            for i in range(num_samples):
                t0 = time.perf_counter()
                y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
                t = time.perf_counter() - t0
                for block in model.transformer.h:
                    block.attn.kv_cache.reset_parameters()

                # responde must be dumped to json file
                response = tokenizer.decode(y).replace(prompt,"").replace("\n", " ").replace("\"", "'").replace("\t"," ").replace("\b"," ").replace("\f"," ").replace("\r"," ").replace("\\","").replace("\/","")

                f.write("{")
                f.write("\"verified_mcq\": \"" + prompt.replace("\n", " ") + "\",")
                f.write("\"evaluation\": \"" + response + "\",")
                f.write("\"evaluation_score\": \"" + response[0] + "\"")
            
        if z == len(prompts)-1:
            f.write("}")
        else:
            f.write("},")

        print("MCQ " + str(z) + "/" + str(len(prompts)) + " done", end="\r")

    f.write("]")
    f.close()

    correct_responses, questions_number = count_correct_responses(file_path=file_path)
    print(f'Correct responses: {correct_responses}/{questions_number}')

    # write correct_responses and questions_number to json file
    f = open(file_path, "r")
    data = f.read()
    f.close()

    f = open(file_path, "w")
    f.write(data[:-1])

    f.write(",{")
    f.write("\"correct_responses\": \"" + str(correct_responses) + "\",")
    f.write("\"functional_programming\": \"" + str(functional_programming) + "\",")
    f.write("\"algorithms\": \"" + str(algorithms) + "\",")
    f.write("\"fundations\": \"" + str(fundations) + "\",")
    f.write("\"abstract_machines\": \"" + str(abstract_machines) + "\",")
    f.write("\"memory_management\": \"" + str(memory_management) + "\",")
    f.write("\"names_and_the_environment\": \"" + str(names_and_the_environment) + "\",")
    f.write("\"describing_a_programming_language\": \"" + str(describing_a_programming_language) + "\",")
    f.write("\"object_oriented_paradigm\": \"" + str(object_oriented_paradigm) + "\",")
    f.write("\"control_structure\": \"" + str(control_structure) + "\",")
    f.write("\"structuring_data\": \"" + str(structuring_data) + "\",")
    f.write("\"programming_languages\": \"" + str(programming_languages) + "\",")
    f.write("\"num_functional_programming\": \"" + str(num_functional_programming) + "\",")
    f.write("\"num_algorithms\": \"" + str(num_algorithms) + "\",")
    f.write("\"num_fundations\": \"" + str(num_fundations) + "\",")
    f.write("\"num_abstract_machines\": \"" + str(num_abstract_machines) + "\",")
    f.write("\"num_memory_management\": \"" + str(num_memory_management) + "\",")
    f.write("\"num_names_and_the_environment\": \"" + str(num_names_and_the_environment) + "\",")
    f.write("\"num_describing_a_programming_language\": \"" + str(num_describing_a_programming_language) + "\",")
    f.write("\"num_object_oriented_paradigm\": \"" + str(num_object_oriented_paradigm) + "\",")
    f.write("\"num_control_structure\": \"" + str(num_control_structure) + "\",")
    f.write("\"num_structuring_data\": \"" + str(num_structuring_data) + "\",")
    f.write("\"num_programming_languages\": \"" + str(num_programming_languages) + "\",")
    f.write("\"questions_number\": \"" + str(questions_number) + "\"")
    f.write("}")
    f.write("]")
    f.close()

def get_verification_prompts(add_prompt:bool=True, file_path:str="data/results/pretrained/Llama-2-7b-chat-hf/base/MCQs_PL_all.json"):
    prompt = []

    init_prompt = ""
    real_answer_prompt = ""
    llm_answer_prompt = ""
    end_prompt = ""

    if add_prompt:
        init_prompt = "You are an expert professor specialized in checking students' answers to questions. You are checking the following question: '"
        real_answer_prompt = "'. Here is the real answer: '"
        llm_answer_prompt = "'. You are checking the following student’s answer: '"
        end_prompt = "...'. What grade do you give, where 0 is incorrect and 1 is correct? Give me only 0 or 1 as response. If the student’s answer is not related to the question, give me 0. Response: "

    functional_programming = 0
    algorithms = 0
    fundations = 0
    abstract_machines = 0
    memory_management = 0
    names_and_the_environment = 0
    describing_a_programming_language = 0
    object_oriented_paradigm = 0
    control_structure = 0
    structuring_data = 0
    programming_languages = 0

    num_functional_programming = 0
    num_algorithms = 0
    num_fundations = 0
    num_abstract_machines = 0
    num_memory_management = 0
    num_names_and_the_environment = 0
    num_describing_a_programming_language = 0
    num_object_oriented_paradigm = 0
    num_control_structure = 0
    num_structuring_data = 0
    num_programming_languages = 0
    with open(file_path) as json_file:
        data = json.load(json_file, strict=False)
        with open("data/MCQs/MCQs_PL_all.json") as json_file2:
            data2 = json.load(json_file2, strict=False)
            for i, mcq in enumerate(data):
                """
                Functional Programming: 1
                Algorithms: 2
                Fundations: 3
                Abstract Machines: 4
                Memory Management: 5
                Names and the Environment: 7
                Describing a Programming Language: 8
                Object-Oriented Paradigm: 11
                Control Structure: 11
                Structuring Data: 29
                Programming Languages: 81
                """
                if add_prompt:
                    prompt.append(init_prompt + mcq["question"].replace("Answer the following question and provide the correct answer. The question is a multiple-choice question with a unique correct answer. ", "").replace(" Answer the question by providing the correct alternative. Do not provide an empty answer. Answer: ", "") + real_answer_prompt + mcq["correct_answers"] + llm_answer_prompt + mcq["answers"][:3] + end_prompt)
                else:
                    correct_answers = mcq["correct_answers"][0]
                    correct_answers = [correct_answer.strip() for correct_answer in correct_answers.split(',')]

                    answer = mcq["answers"]
                    result = []
                    tmp_length = len(answer)
                    valid_chars = {' ', '.'}
                    t = 0
                    while t < tmp_length:
                        # Check if the current character is one of the target letters
                        if answer[t] in {'a', 'b', 'c', 'd'}:
                            # Check if the character before is either nothing, a space, or a dot
                            if t == 0 or answer[t - 1] in valid_chars:
                                # Check if the character after is a closing parenthesis
                                if t + 1 < tmp_length and answer[t + 1] == ')':
                                    if t + 2 < tmp_length and answer[t + 2] in valid_chars:
                                        result.append(answer[t])
                        t += 1

                    # if at least one of the correct answers is in the result
                    tmp = False
                    for correct_answer in correct_answers:
                        if correct_answer in result:
                            tmp = True
                    
                    if tmp:
                        prompt.append("1")
                        if data2[i]["topic"] == "Functional Programming":
                            functional_programming += 1
                        elif data2[i]["topic"] == "Algorithms":
                            algorithms += 1
                        elif data2[i]["topic"] == "Fundations":
                            fundations += 1
                        elif data2[i]["topic"] == "Abstract Machines":
                            abstract_machines += 1
                        elif data2[i]["topic"] == "Memory Management":
                            memory_management += 1
                        elif data2[i]["topic"] == "Names and the Environment":
                            names_and_the_environment += 1
                        elif data2[i]["topic"] == "Describing a Programming Language":
                            describing_a_programming_language += 1
                        elif data2[i]["topic"] == "Object-Oriented Paradigm":
                            object_oriented_paradigm += 1
                        elif data2[i]["topic"] == "Control Structure":
                            control_structure += 1
                        elif data2[i]["topic"] == "Structuring Data":
                            structuring_data += 1
                        elif data2[i]["topic"] == "Programming Languages":
                            programming_languages += 1
                    else:
                        prompt.append("0")

                    if data2[i]["topic"] == "Functional Programming":
                        num_functional_programming += 1
                    elif data2[i]["topic"] == "Algorithms":
                        num_algorithms += 1
                    elif data2[i]["topic"] == "Fundations":
                        num_fundations += 1
                    elif data2[i]["topic"] == "Abstract Machines":
                        num_abstract_machines += 1
                    elif data2[i]["topic"] == "Memory Management":
                        num_memory_management += 1
                    elif data2[i]["topic"] == "Names and the Environment":
                        num_names_and_the_environment += 1
                    elif data2[i]["topic"] == "Describing a Programming Language":
                        num_describing_a_programming_language += 1
                    elif data2[i]["topic"] == "Object-Oriented Paradigm":
                        num_object_oriented_paradigm += 1
                    elif data2[i]["topic"] == "Control Structure":
                        num_control_structure += 1
                    elif data2[i]["topic"] == "Structuring Data":
                        num_structuring_data += 1
                    elif data2[i]["topic"] == "Programming Languages":
                        num_programming_languages += 1
    return prompt, functional_programming, algorithms, fundations, abstract_machines, memory_management, names_and_the_environment, describing_a_programming_language, object_oriented_paradigm, control_structure, structuring_data, programming_languages, num_functional_programming, num_algorithms, num_fundations, num_abstract_machines, num_memory_management, num_names_and_the_environment, num_describing_a_programming_language, num_object_oriented_paradigm, num_control_structure, num_structuring_data, num_programming_languages

def count_correct_responses(file_path='data/verified_results/pretrained/Llama-2-7b-chat-hf/base/MCQs_PL_all.json'):
    with open(file_path, 'r') as f:
        verified_results = json.load(f, strict=False)
    correct_responses = 0
    for result in verified_results:
        int_to_add = result['evaluation_score']
        # if int_to_add is convertible to int, add it to correct_responses
        try:
            int_to_add = int(int_to_add)
        except:
            int_to_add = 0
        correct_responses += int_to_add
    questions_number = len(verified_results)
    
    return correct_responses, questions_number

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
