from langchain_community.tools import tool


@tool
def multiply(a:int, b:int)->int:
    "Multiply two numbers"
    return a*b

@tool
def add(a:int, b:int)->int:
    "sum two numbers"
    return a+b

class MathToolKit:
    def get_tools(self):
        return [add, multiply]
    

toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=> ", tool.description)
    print(tool.invoke({'a':10, 'b': 34}))