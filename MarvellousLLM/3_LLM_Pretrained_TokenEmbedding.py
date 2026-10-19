import torch 

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# pretrained model name
model_name = "distilgpt2" 

# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name) 

# Load pretrained Language model 
model = AutoModelForCausalLM.from_pretrained(model_name) 

# Input sentence 
sentence = "I love AI" 

# Tokenize the sentence first, then convert tokens to PyTorch tensors (numeric matrix)
inputs = tokenizer(
    sentence,
    return_tensors = "pt" 
)

# Extract token ID
token_ids = inputs["input_ids"] 

# Access the pretrained input embedding layer 
embedding_layer = model.get_input_embeddings() 

# Retriev the embedding vectors 
embeddings = embedding_layer(token_ids) 

print("Input sentence : ", sentence) 
print("Token Ids : ", token_ids) 

print("Embedding shape : ")  
print(embeddings.shape)

print("First token embedding : ")
print(embeddings[0, 0:10]) 

# for type checking 
print(type(tokenizer).mro()) 
