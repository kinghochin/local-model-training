from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./custom-llm")
tokenizer = AutoTokenizer.from_pretrained("./custom-llm")

input_text = "How do I get a refund?"
# Tokenize and create attention mask
encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, do_sample=True)
print(f"Result: {tokenizer.decode(output[0], skip_special_tokens=True)}")

