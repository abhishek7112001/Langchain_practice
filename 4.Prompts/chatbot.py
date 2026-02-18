from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from dotenv import load_dotenv

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# very important: Now to maintain the chat history, i need to create a list and append the user_input, result of the model
chat_history=[
    SystemMessage("You are an AI assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input =='exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)