import json
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.deepmath import is_equiv


def generate_results(ds, dump_path):
    prompt_list = [data[0]['content'] for data in ds['prompt']]
    solution_list = [data for data in ds['final_answer']]

    outputs = llm.generate(prompts=prompt_list, sampling_params=sampling_params)
    generated_text_list = [output.outputs[0].text for output in outputs]
    answer_list = [extract_solution(text) for text in generated_text_list]
    results = []

    scores = 0
    for prompt, generated_text, answer, solution in zip(prompt_list, generated_text_list, answer_list, solution_list):
        score = int(is_equiv(answer, solution))
        results.append(
            {
                'prompt': prompt,
                'generated_text': generated_text,
                'generated_answer': answer,
                'ground_truth': solution,
                'score': score
            }
        )
        scores += score
    with open(dump_path, 'w') as fp:
        json.dump(results, fp, indent=4)

    print(f"scores: {scores}, total: {len(prompt_list)}, average: {scores / len(prompt_list)}")

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

MODEL_PATH = "/root/data/models/Qwen2.5-1.5B-Instruct-GRPO-deepmath-1k"
TRAIN_DUMP_PATH = "/root/data/code/math-ocr-zero/examples/results/Qwen2.5-1.5B-Instruct-GRPO-deepmath-1k.train.json"
TEST_DUMP_PATH = "/root/data/code/math-ocr-zero/examples/results/Qwen2.5-1.5B-Instruct-GRPO-deepmath-1k.test.json"

llm = LLM(model=MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=2048,
)
train_ds = load_dataset("/root/data/deepmath-1000", split='train')
test_ds = load_dataset("/root/data/deepmath-1000", split='test')
print("====== train dataset ======")
generate_results(train_ds, TRAIN_DUMP_PATH)
print("====== test dataset ======")
generate_results(test_ds, TEST_DUMP_PATH)

