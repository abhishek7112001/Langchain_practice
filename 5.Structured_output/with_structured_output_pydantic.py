from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

class Review(BaseModel):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke('PySpark is the Python API for Apache Spark, a framework for large-scale data processing that enables distributed computing across clusters of machines. It allows users to perform big data analytics and machine learning on datasets of all sizes by combining Python simplicity with Spark processing power')

print(result)
print(result.summary)
print(result.sentiment)

