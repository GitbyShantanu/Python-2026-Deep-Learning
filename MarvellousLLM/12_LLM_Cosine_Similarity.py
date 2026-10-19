from sentence_transformers import SentenceTransformer 
from sentence_transformers import util 

# Load pretrained embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "Python is a programming language"
sentence2 = "Python is used for software development",

embedding1 = model.encode(sentence1, convert_to_tensor=True)  
embedding2 = model.encode(sentence2, convert_to_tensor=True)  

# Calculate cosine similarity 
similarity = util.cos_sim(embedding1, embedding2) 

print("Sentence 1 : ", sentence1) 
print("Sentence 2 : ", sentence2)  

print("Cosine similarity : ") 
print(similarity.item()) 