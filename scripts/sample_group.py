from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from r1.data.countdown import compute_score, get_dataset

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
model = LLM(
    model=model_name,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
dataset = get_dataset("train", True)
chats = [d["prompt"] for d in dataset[:500]]

outputs = model.chat(
    chats[:5_000],
    sampling_params=SamplingParams(
        max_tokens=500,
        n=5,
        # 0 allows us to get logprobs of the sampled token
        # prompt_logprobs=0,
        # logprobs=0,
    ),
)

scores = []
for output, row in zip(outputs, dataset):
    responses = [o.text for o in output.outputs]
    scores.append(
        [
            compute_score(
                r,
                row["task"],
                fmt_score=0.01,
                score=1,
            )
            for r in responses
        ]
    )
len([rs for rs in scores if any([r == 1 for r in rs])])

# batch_size = 5_000
# all_outputs = []
# for i in tqdm(range(0, len(chats), batch_size), total=len(chats) // batch_size):
#     batch_chats = chats[i : i + batch_size]
#     outputs = model.chat(
#         batch_chats,
#         sampling_params=SamplingParams(
#             max_tokens=500,
#             n=8,
#             # 0 allows us to get logprobs of the sampled token
#             prompt_logprobs=0,
#             logprobs=0,
#         ),
#     )
#     all_outputs.extend(outputs)
