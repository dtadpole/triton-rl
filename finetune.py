from huggingface_hub import login

hf_token = "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"
login(hf_token)

import wandb

# wb_token = "your_wandb_token"
# wandb.login()
# run = wandb.init(
#     project='Fine-tune-DeepSeek-R1-Distill-Qwen-14B on Legal COT Dataset',
#     job_type="training",
#     anonymous="allow"
# )

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


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thought to ensure a logical and accurate response.

### Instruction:
You are a legal expert with advanced knowledge in legal reasoning, case analysis, and interpretation of laws.
Please answer the following legal question.
### Question:
{}
### Response:
<think>
{}
</think>
{}"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input_text, cot, output_text in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input_text, cot, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

from datasets import load_dataset
dataset = load_dataset("kienhoang123/QR-legal", "en", split="train[0:500]", trust_remote_code=True)
dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset["text"][0])

####Note: This is a pseudo dataset. Please create or use an appropriate dataset for your own use case.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()


question = "A contract was signed between two parties, but one party claims they were under duress. What legal principles apply to determine the contract’s validity?"


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


new_model_local = "DeepSeek-R1-Legal-COT"
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)
model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")
