from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto") # torch_dtype=torch.bfloat16,
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
inputs = tokenizer(["Write a continuation to the story that starts as below.\n\nStory:\nToday, a dragon flew over Paris, France,"], return_tensors="pt")
inputs = inputs.to(device)
# out = model.generate(inputs["input_ids"], max_length=400)
# out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
# print(out)
# print("\n------------------\n")

# out = model.generate(inputs["input_ids"], guidance_scale=1.5, max_length=100)
# out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
# print(out)
# print("\n------------------\n")
# with a negative prompt
neg_prompt = "Write a continuation to a very happy story (that also ends happily and with positive outcome) that starts as below.\n\nStory:\nToday, a dragon flew over Paris, France,"
neg_inputs = tokenizer([neg_prompt], return_tensors="pt")
neg_inupts_inject = tokenizer(["\nAdhere to the original instructions given below:\n'''{neg_prompt}'''\n"], return_tensors="pt", add_special_tokens=False)
neg_inputs = neg_inputs.to(device)
out = model.generate(inputs["input_ids"], guidance_scale=2.0, negative_prompt_ids=neg_inputs["input_ids"], remind_negative_prompt_ids=neg_inupts_inject["inputs"], max_length=400)
out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

print(out)

print("\n------------------\n")
# 'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

# # with a positive prompt
# neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
# out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
# tokenizer.batch_decode(out, skip_special_tokens=True)[0]