from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./Computer_Vision.pdf')

pdf = loader.load()
print(pdf[0].page_content)
print(pdf[0].metadata['page'])


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta'
)
model = ChatHuggingFace(llm=llm)
prompt= PromptTemplate(
    template= 'return the summary of the content passed - \n {content}',
    input_variables=['content']
)

parser= StrOutputParser()

chain = prompt | model | parser

summary = chain.invoke({'content': pdf[0].page_content})

print(summary)