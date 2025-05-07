import base64
import json

from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from verl.utils.reward_score.deepmath import is_equiv, extract_solution

MODEL_PATH = "/root/data/models/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/root/data/code/math-ocr-zero/data/deepmath-ocr-1000/"

TRAIN_DUMP_PATH = "/root/data/code/math-ocr-zero/examples/results/Qwen2.5-VL-3B-Instruct.train.json"
TEST_DUMP_PATH = "/root/data/code/math-ocr-zero/examples/results/Qwen2.5-VL-3B-Instruct.test.json"


def generate_results(ds, dump_path):
    prompt_list = [data[0]['content'] for data in ds['prompt']]
    image_list = [data[0] for data in ds['images']]
    solution_list = [data for data in ds['final_answer']]

    llm_inputs_list = []
    for prompt, image in zip(prompt_list, image_list):
        llm_inputs = get_llm_inputs(prompt, image)
        llm_inputs_list.append(llm_inputs)
    outputs = llm.generate(prompts=llm_inputs_list, sampling_params=sampling_params)
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


def get_llm_inputs(prompt, image):
    # encoded_image = base64.b64encode(image)
    # encoded_image_text = encoded_image.decode("utf-8")
    # base64_qwen = f"data:image;base64,{encoded_image_text}"

    image_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
            ],
        },
    ]

    image_inputs, video_inputs = process_vision_info(image_message)

    prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs},
    }
    return llm_inputs


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1,
    max_tokens=2048,
    stop_token_ids=[],
)

train_ds = load_dataset(DATA_PATH, split='train')
test_ds = load_dataset(DATA_PATH, split='test')

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

print("====== train dataset ======")
generate_results(train_ds, TRAIN_DUMP_PATH)
print("====== test dataset ======")
generate_results(test_ds, TEST_DUMP_PATH)
