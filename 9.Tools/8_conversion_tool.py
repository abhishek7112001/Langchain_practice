from langchain_community.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env into environment variables

@tool
def get_conversion_factor(base_curr: str, target_curr:str)->float:
    """
        This function fetches the currency conversion rate between base and target currency
    """
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_curr}/{target_curr}"
    response = requests.get(url)
    return response.json()["conversion_rate"]

@tool 
def conversion(base_curr_val:int, conversion_rate:Annotated[float, InjectedToolArg])->float:
    '''
        This function returns the value of target currency given the base_currency_value and conversion rate
    '''
    return conversion_rate*base_curr_val

# print(get_conversion_factor.invoke({'base_curr':'USD', 'target_curr': 'INR'}))
# print(conversion.invoke({'base_curr_val':10, 'conversion_rate': 90.3383}))


# llm = HuggingFaceEndpoint(
#     repo_id = "meta-llama/Llama-3.1-8B-Instruct",
#     temperature=0,
#     max_new_tokens=512
# )

# model = ChatHuggingFace(llm=llm)

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
)

model_with_tools = model.bind_tools([get_conversion_factor, conversion])  # wrapped the model with tools

messages = [HumanMessage('what is the conversion rate between USD and INR and based on that what will be the conversion of 12USD into INR')]

ai_message = model_with_tools.invoke(messages) # first request
messages.append(ai_message)

# print(ai_message.tool_calls)
for tool_call in ai_message.tool_calls:
    if tool_call['name']=='get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate= float(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)
        
    if tool_call['name']=='conversion':
        tool_call['args']['conversion_rate']= conversion_rate
        tool_message2 = conversion.invoke(tool_call)
        messages.append(tool_message2)



# final invoke
print(model_with_tools.invoke(messages))

