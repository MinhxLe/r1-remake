config = GRPOConfig()
trainer = CountdownGRPO(config)

training_batch = trainer.train_dataset.select(range(config.train_batch_size))

# implicit iteration over elements of training_batch
batch_response_groups = []
batch_rewards = []
batch_advantages = []
# iteration over all prompts holding pi_theta_old model constant
# potentially could relax this and let theta_old vary with task
task = training_batch[0]
response_group = trainer._generate_response_group(
    task["prompt"][0]["content"], 
    trainer.config.group_size,
    trainer.config.max_new_tokens,
)

batch_response_groups.append(response_group)
rewards = trainer._compute_rewards(response_group.responses, task)
batch_rewards.append(rewards)
batch_advantages.append(trainer._normalize_advantages(rewards))

new_log_probs = trainer._compute_log_probs(response_group)
#not there yet - these are too different!

input_ids = trainer.tokenizer("Tell me a joke please!", return_tensors="pt").input_ids.to(trainer.device)
with torch.no_grad():
    outputs = trainer.model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        pad_token_id=trainer.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True
    )

    response_ids = outputs.sequences[0]
    print(trainer.tokenizer.decode(response_ids, skip_special_tokens=True))
    
    scores = torch.stack(outputs.scores)  # [new_tokens, 1, vocab_size]