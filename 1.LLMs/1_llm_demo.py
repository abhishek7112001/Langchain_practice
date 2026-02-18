# from langchain_openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = OpenAI(model="gpt-3.5-turbo-instruct")

# result = llm.invoke("Who is the PM of India?")

# print(result)



from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# 1. Load the environment variables from the .env file.
# This should contain the GEMINI_API_KEY.
load_dotenv()

# Optional: Set the API Key programmatically if it's not in the .env file
# or if you want to explicitly check for it.
# os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE' 

# 2. Use ChatGoogleGenerativeAI (or GoogleGenerativeAI) instead of OpenAI.
# 'gemini-2.5-flash' is the recommended model for general tasks.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Use the invoke method to call the model
result = llm.invoke("Who is the PM of India?")

# 4. Print the result (the content of the response)
print(result.content)