from langchain_community.document_loaders import TextLoader
# now lets summarize the txt using LLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta"
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template = 'Write the Summary of the poem - \n {poem}',
    input_variables = ['poem']
)
parser = StrOutputParser()



loader = TextLoader('train.txt', encoding='utf-8')
docs = loader.load()

# print(docs)
# print(docs[0])
# print(docs[0].metadata)
# print(docs[0].metadata)
print(docs[0].page_content)

chain = prompt | model | parser

summary = chain.invoke({'poem': docs[0].page_content})

print(summary)