from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

dataset = load_dataset("csv", data_files="faq.csv")
dataset = dataset["train"].train_test_split(test_size=0.1)

model_name = "gpt2"  # You can also try EleutherAI or TinyLLaMA
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(example):
    tokenized = tokenizer(
        example["input"] + tokenizer.eos_token + example["output"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # 👈 Important line
    return tokenized


tokenized_dataset = dataset.map(tokenize)

model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

print('Starting the training...')
trainer.train()

print('Training Completed!')
model.save_pretrained("./custom-llm")
tokenizer.save_pretrained("./custom-llm")

