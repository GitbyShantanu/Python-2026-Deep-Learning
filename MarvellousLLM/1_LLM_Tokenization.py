from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Load Pretrained tokenizer 
tokenizer : PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2") 

# Input sentence 
sentence = " I Love AI" 

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