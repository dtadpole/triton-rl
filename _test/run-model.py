from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 2048
lora_rank = 16

# get deepseek finetuned qwen model
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "Qwen/Qwen2.5-7B-Instruct",
    # model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ], # Remove QKVO if out of memory
#     lora_alpha = lora_rank,
#     use_gradient_checkpointing = "unsloth", # Enable long context finetuning
#     random_state = 3407,
# )

# run model
# model.generate("Hello, how are you?")

text = tokenizer.apply_chat_template([
    # {"role" : "user", "content" : "How many r's are in strawberry?"},
    {"role" : "user", "content" : "Which is bigger 9.8 or 9.12?"},
], tokenize = False, add_generation_prompt = True)


from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = max_seq_length,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

print('##########################')
print(output)
