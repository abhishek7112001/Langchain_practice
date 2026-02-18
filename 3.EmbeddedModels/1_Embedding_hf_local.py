from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vector= embedding.embed_query("What is Data Science?")

print(str(vector))

