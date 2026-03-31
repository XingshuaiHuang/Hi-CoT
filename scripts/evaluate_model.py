import json
import os
import time

import fire
import numpy as np
import vllm
from functools import partial
from transformers import AutoTokenizer

from datasets import load_from_disk
from utils.math_grader import (answer_tag_reward_fn,
                                   r1_distill_qwen_math_reward_fn,
                                   hierarchy_format_reward,
                                    boxed_reward_fn)


HIERARCHICAL_REASONING_PROMPT = """
    You are a reasoning assistant that solves problems by alternating between <|instruction|> (planning what to do next) and <|execution|> (executing that plan).  
    Each <|instruction|> describes the reasoning step or plan, and each <|execution|> performs the corresponding reasoning or computation.

    You should start with an instruction and follow this format strictly:
    <|instruction|> Step 1: ...
    <|execution|> Step 1: ...
    <|instruction|> Step 2: ...
    <|execution|> Step 2: ...
    ...
    Finally, put your final answer within \\boxed{}.
    """


HICOT_WO_STRUCTURE_PROMPT = """
    You are a reasoning assistant that solves problems by alternating between <|instruction|> (planning what to do next) and <|execution|> (executing that plan).  
    Each <|instruction|> describes the reasoning step or plan, and each <|execution|> performs the corresponding reasoning or computation.
    Finally, put your final answer within \\boxed{}.
    """


COT_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}.\n"


STANDARD_PROMPT = "Answer the question directly and put your final answer within \\boxed{}.\n"


PLAN_SOLVE_PROMPT = """
    Let’s first understand the problem, extract relevant variables and their corresponding numerals, and
    make a complete plan.Then, let’s carry out the plan, calculate intermediate variables (pay attention to
    correct numerical calculation and commonsense), solve the problem step by step, and finally put your final answer within \\boxed{}.
    """


def apply_prompt_template(model_name: str, prompt: str, question: str):
    if prompt == "cot":
        prompt_template = COT_PROMPT
    elif prompt == "hicot":
        prompt_template = HIERARCHICAL_REASONING_PROMPT
    elif prompt == "hicot_wo_structure":
        prompt_template = HICOT_WO_STRUCTURE_PROMPT
    elif prompt == "ps":
        prompt_template = PLAN_SOLVE_PROMPT
    elif prompt == "standard":
        prompt_template = STANDARD_PROMPT
    else:
        raise ValueError("Unknown prompt: " + prompt)
    
    if model_name.startswith("Qwen/Qwen"):
        final_prompt = (
            f"<|im_start|>system\n{prompt_template}<|im_end|>\n<|im_start|>user\n"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    elif model_name.startswith("deepseek-ai/DeepSeek-R1-Distill"):
        final_prompt = (
            f"<｜begin▁of▁sentence｜><｜User｜>{question}\n{prompt_template}<｜Assistant｜><think>\n"
        )
    else:
        raise ValueError("Unknown model: " + model_name)
    
    return final_prompt


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    template: str = "qwen_math",
    dataset_name: str = "./data/evaluation_suite",
    temperature: float = 0,
    top_p: float = 1,
    max_tokens: int = 4096,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 1,
    max_test: int = 999999,
    save: bool = True,
    gpu_memory_utilization: float = 0.98,
):

    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # logprobs=2,
        seed=int(time.time_ns()),
    )

    model = vllm.LLM(
        model_name,
        # swap_space=32,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print("Using template:", template)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if model_name.startswith("Qwen/Qwen"):
        math_reward_fn = boxed_reward_fn
    elif model_name.startswith("deepseek-ai/DeepSeek-R1-Distill"):
        math_reward_fn = r1_distill_qwen_math_reward_fn
    else:
        raise ValueError("Unknown model: " + model_name)
    
    apply_template = partial(apply_prompt_template, model_name, template)
    
    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    results_with_correct_hierarchy_format = {}
    avg_lens_with_correct_hierarchy_format = {}
    number_of_correct_hierarchy_format = {}
    total_number_of_responses = {}
    to_be_saved = []
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue
        prompts = dataset["problem"][:max_test]
        targets = dataset["answer"][:max_test]

        prompts = list(map(apply_template, prompts))
        print("inference for ", task_name)
        outputs = model.generate(prompts, sampling_params)
        batch_scores = []
        batch_formatted = []
        batch_lengths = []
        rewards_with_correct_hierarchy_format = []
        number_of_correct_hierarchy_format[task_name] = 0.0
        total_number_of_responses[task_name] = 0.0
        for k in range(len(outputs)):
            output = outputs[k]
            gt_repeated = [targets[k]] * sampling_params.n
            rewards, infos = [], []
            for model_output, gt in zip(output.outputs, gt_repeated):
                info, r = math_reward_fn(model_output.text, gt, fast=False)
                format_r = hierarchy_format_reward(model_output.text)
                rewards.append(r)
                infos.append(info)
                total_number_of_responses[task_name] += 1.0
                if format_r == 1:
                    rewards_with_correct_hierarchy_format.append(r)
                    number_of_correct_hierarchy_format[task_name] += 1.0
                    token_len = len(model_output.token_ids)
                    avg_lens_with_correct_hierarchy_format.setdefault(task_name, []).append(token_len)
            rewards = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards.mean())

            if infos[0]:
                batch_formatted.append(np.array([i["formatted"] for i in infos]).sum())

            to_be_saved.append(
                {
                    "task_name": task_name,
                    "prompt": output.prompt,
                    "gt": gt_repeated,
                    "model_output": [o.text for o in output.outputs],
                    "model_output_token_ids": [o.token_ids for o in output.outputs],
                    "reward": [r for r in rewards],
                }
            )

        results[task_name] = np.mean(batch_scores)
        avg_lens[task_name] = np.mean(batch_lengths)
        if batch_formatted:
            formatted[task_name] = np.mean(batch_formatted)
        max_lens[task_name] = np.max(batch_lengths)
        if rewards_with_correct_hierarchy_format:
            results_with_correct_hierarchy_format[task_name] = np.mean(rewards_with_correct_hierarchy_format)
        if avg_lens_with_correct_hierarchy_format:
            avg_lens_with_correct_hierarchy_format[task_name] = np.mean(avg_lens_with_correct_hierarchy_format[task_name])

    print("\naccuracy:", results)
    print("\navg accuracy:", np.mean(list(results.values())))
    print("\nresults with correct hierarchy format:", results_with_correct_hierarchy_format)
    print("\nnumber of correct hierarchy format:", number_of_correct_hierarchy_format)
    print("\ntotal number of responses:", total_number_of_responses)
    print("\navg_lens:", avg_lens)
    print("\navg_lens_with_correct_hierarchy_format:", avg_lens_with_correct_hierarchy_format)
    print("\nmax_lens:", max_lens)
    print("\nformatted:", formatted)

    if save:
        val_folder = "model_eval_outputs"
        os.makedirs(val_folder, exist_ok=True)
        
        # fn = "model_eval_out_" + model_name.replace("/", "_") + str(int(time.time()))
        # fn = f"{val_folder}/{fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        # print(f"saving model outputs at {fn}")
        # json.dump(
        #     to_be_saved,
        #     open(
        #         fn,
        #         "w",
        #     ),
        #     indent=4,
        # )
        
        # Save the overall results as well.
        overall_results = {
            "results": results,
            "avg": np.mean(list(results.values())),
            "avg_lens": avg_lens,
            "results_with_correct_hierarchy_format": results_with_correct_hierarchy_format,
            "avg_lens_with_correct_hierarchy_format": avg_lens_with_correct_hierarchy_format,
            "number_of_correct_hierarchy_format": number_of_correct_hierarchy_format,
            "total_number_of_responses": total_number_of_responses,
        }
        overall_fn = "summary_" + model_name.replace("/", "_")
        
        overall_fn = f"{val_folder}/{overall_fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        print(f"saving overall results at {overall_fn}")
        json.dump(
            overall_results,
            open(
                overall_fn,
                "w",
            ),
            indent=4,
        )


fire.Fire(main)