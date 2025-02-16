# import torch
# from huggingface_hub import login
# hf_token = "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"
# login(hf_token)

import torch
from unsloth import FastLanguageModel

# from unsloth import FastLanguageModel

# Check CUDA compatibility
major_version, minor_version = torch.cuda.get_device_capability()

# Instantiate FastLanguageModel
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2b-it",
    max_seq_length=4096,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

prompt = """Based on given instruction and context, generate an appropriate response

### Instruction:
{}

### Context:
{}

### Response:
{}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    contexts = examples["context"]
    responses = examples["response"]
    texts = []

    for i,j,k  in zip(instructions, contexts,responses):
        text = prompt.format(i,j,k) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 2,
        max_steps = 10,
        learning_rate = 0.0005,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Training
trainer_stats = trainer.train()

inputs = tokenizer(
    [
        prompt.format(
            "Provide a detailed explanation of the events and circumstances that led to the outbreak of World War II.",  # instruction
            " The goal is to offer a clear and informative account of the factors, political decisions, and international tensions that played a crucial role in triggering World War II. Ensure that the explanation covers the period leading up to the war, key events, and the involvement of major nations",  # context
            " ",  # response
        )
    ] * 1,
    return_tensors="pt",
).to("cuda")

# Generate response
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1028)

# Output:
# The outbreak of World War II was triggered by a series of events that began
# in the early 1930s. The Great Depression of the 1930s led to a decline in global economic activity,
# which in turn led to a decline in trade and a rise in nationalist sentiment.
# This led to a series of diplomatic tensions between nations, particularly between Germany and Britain.
# Germany's aggressive foreign policy and expansionist ambitions led to a series of diplomatic
# incidents and military provocations, culminating in the invasion of Poland in September 1939.
# This led to the outbreak of World War II, with Germany and Britain declaring war
# on Poland and France declaring war on Germany.
# The war quickly escalated into a global conflict, with nations from across the world involved in the conflict.
# The war was characterized by massive military mobilization, mass destruction, and the use of new weapons such as atomic bombs and nuclear weapons.
# The war ended in 1945 with the defeat of Germany and Japan, and the establishment of the United Nations as a global organization to prevent future conflicts.
