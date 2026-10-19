from sentence_transformers import SentenceTransformer 

# Load pretrained embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language",
    "Python is used for data science",
    "I support indian cricket team" 
]

embeddings = model.encode(sentences) 

# Display sentence embeddings 
for index, embedding in enumerate(embeddings):
    print("-"*40) 
    print("\nSentence : ", sentences[index]) 

    print("Embedding : ")
    print(embedding[:10]) 

    print("Total dimention : ", len(embedding)) 