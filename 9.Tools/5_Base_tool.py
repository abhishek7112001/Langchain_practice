from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a:int = Field(required=True, description="The First number to add")
    b:int = Field(required=True, description="The Second number to add")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a:int, b:int)->int:
        return a*b
    

multiply_tool= MultiplyTool()
result = multiply_tool.invoke({'a':8, 'b': 8})
print(result)
    