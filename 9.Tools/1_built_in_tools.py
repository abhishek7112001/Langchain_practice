from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke('delhi AQI')
print(result)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)