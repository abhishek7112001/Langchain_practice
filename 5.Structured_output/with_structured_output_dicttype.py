from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
from dotenv import load_dotenv
from typing import TypedDict
# import os

load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure your API key is loaded

# genai.configure(api_key=GOOGLE_API_KEY)

# print("Available models:")
# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         print(f"- {m.name}")
model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke('PySpark is the Python API for Apache Spark, a framework for large-scale data processing that enables distributed computing across clusters of machines. It allows users to perform big data analytics and machine learning on datasets of all sizes by combining Python simplicity with Spark processing power')

print(result)
print(result['summary'])
print(result['sentiment'])

