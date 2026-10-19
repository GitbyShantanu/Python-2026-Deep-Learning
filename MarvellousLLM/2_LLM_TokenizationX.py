from transformers import AutoTokenizer

# Load Pretrained tokenizer 
tokenizer = AutoTokenizer.from_pretrained("gpt2") 

# Input sentence 
sentence = "Python is easy programming language I love it" 

# Convert sentence into tokens 
tokens = tokenizer.tokenize(sentence) 

token_ids = tokenizer.encode(
    sentence,
    add_special_tokens = False 
)

# Convert token ids back to text 
decoded = tokenizer.decode(token_ids)

print("Original Sentence : ", sentence)  

print("Tokens : ", tokens) 

print("Token IDs : ", token_ids) 

print("Decoded sentence : ", decoded) 