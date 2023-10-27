from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
# out = model.generate(inputs["input_ids"], guidance_scale=1.5)
# tokenizer.batch_decode(out, skip_special_tokens=True)[0]
# 'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

# with a negative prompt
neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"], max_length=100)
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
# 'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

# # with a positive prompt
# neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
# out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
# tokenizer.batch_decode(out, skip_special_tokens=True)[0]