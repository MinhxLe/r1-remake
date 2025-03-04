import torch

from r1.data.core import Chat


@torch.no_grad()
def sample_group(self, model, prompt: list[Chat]) -> Group:
    tokenizer, cfg = self.tokenizer, self.cfg
    group_size, max_new_tokens = cfg.group_size, cfg.max_new_tokens

    # Initialize vLLM if not already done
    if not hasattr(self, "vllm_engine"):
        self.vllm_engine = LLM(
            model=model.config._name_or_path,  # assuming model has this attribute
            dtype="auto",
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,  # adjust based on your GPU setup
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=cfg.temperature if hasattr(cfg, "temperature") else 1.0,
            top_p=cfg.top_p if hasattr(cfg, "top_p") else 1.0,
            n=group_size,  # number of generations per prompt
        )

    # Format prompt for vLLM
    if cfg.use_instruct_prompt:
        formatted_prompt = tokenizer.apply_chat_template([prompt], return_tensors="pt")
        # Convert to string for vLLM
        formatted_prompt = tokenizer.decode(
            formatted_prompt[0], skip_special_tokens=False
        )
    else:
        formatted_prompt = prompt

    # Generate responses using vLLM
    outputs = self.vllm_engine.generate([formatted_prompt], self.sampling_params)

    # Process the outputs
    prompt_token_ids = tokenizer([formatted_prompt], return_tensors="pt").input_ids
    prompt_length = prompt_token_ids.shape[-1]

    # Extract generated sequences, convert to token IDs
    generations = [output.outputs[0].text for output in outputs]
    all_token_ids = []
    all_logprobs = []
    max_length = 0

    # Process each generated output
    for gen in generations:
        gen_tokens = tokenizer(gen, return_tensors="pt").input_ids[0]
        all_token_ids.append(gen_tokens)
        max_length = max(max_length, len(gen_tokens))

    # Pad sequences to the same length
    response_token_ids = torch.full((group_size, max_length), tokenizer.pad_token_id)
    response_masks = torch.zeros((group_size, max_length), dtype=torch.bool)

    for i, tokens in enumerate(all_token_ids):
        response_token_ids[i, : len(tokens)] = tokens
        response_masks[i, : len(tokens)] = True

    # NOTE: vLLM doesn't directly provide token logprobs in the same way
    # You'll need to decide how to handle this - options include:
    # 1. Run a separate forward pass to get logprobs
    # 2. Use a placeholder and update your downstream code
    # Here's an approach with a separate forward pass:

    response_log_probs = torch.zeros_like(response_token_ids, dtype=torch.float)

    # If you need exact logprobs, you can do this (computationally expensive):
    for i in range(group_size):
        # Get only the valid tokens (not padding)
        valid_tokens = response_token_ids[i, response_masks[i]]

        inputs = torch.cat([prompt_token_ids[0], valid_tokens[:-1]])
        with torch.no_grad():
            outputs = model(inputs.unsqueeze(0))
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, -1)

        # For each position, get the log prob of the actual next token
        for j in range(len(valid_tokens)):
            pos = j + prompt_length - 1 if j > 0 else prompt_length - 1
            if pos < log_probs.shape[1]:
                token_id = valid_tokens[j]
                response_log_probs[i, j] = log_probs[0, pos, token_id]

    return Group(
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_token_ids,
        response_log_probs=response_log_probs,
        response_masks=response_masks,
    )
