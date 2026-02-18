from langchain_community.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

@tool
def multiply(a: int, b: int)-> int:
    "Multiply two numbers"
    return a*b


llm = HuggingFaceEndpoint(
    repo_id='unsloth/gemma-3-27b-it-GGUF'
)

model = ChatHuggingFace(llm=llm)
model_with_tool = model.bind_tools([multiply])

query =HumanMessage('Multiply 8 with 3')
messages=[query]


# print(model_with_tool)

# tool calling
result = model_with_tool.invoke(messages)
messages.append(result)
model_with_tool.invoke(messages).content
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)
ans=model_with_tool.invoke(messages)

print(ans)