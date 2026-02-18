# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# result = model.invoke("What is Physics?")

# print(result.content)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# This is the correct, stable, high-quota model
model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

result = model.invoke("What is Physics?")
print(result.content)