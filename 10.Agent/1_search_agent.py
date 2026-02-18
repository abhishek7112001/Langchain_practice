from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain.agents import AgentExecutor
from langchain.agents import create_agent
from langchain import hub


import requests

search_tool = DuckDuckGoSearchRun()
# print(search_tool.invoke("What is the current temperature of noida sector 62"))


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
model = ChatHuggingFace(llm=llm)


prompt = hub.pull('hwchase17/react')

agent = create_agent(
    llm=model,
    tools = [search_tool],
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools= [search_tool],
    verbose = True
)

response = agent_executor.invoke({"input": "Three ways to go Hyderabad from Delhi"})
print(response)
print(response['output'])
