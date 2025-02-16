from huggingface_hub import login

hf_token = "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"
login(hf_token)

import wandb

# wb_token = "your_wandb_token"
wandb.login()
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Qwen-14B on Legal COT Dataset',
    job_type="training",
    anonymous="allow"
)

from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token,
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thought to ensure a logical and accurate response.

### Instruction:
You are a legal expert with advanced knowledge in legal reasoning, case analysis, and interpretation of laws.
Please answer the following legal question.
### Question:
{}
### Response:
<think>{}"""

question = "A contract was signed between two parties, but one party claims they were under duress. What legal principles apply to determine the contract's validity?"
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
