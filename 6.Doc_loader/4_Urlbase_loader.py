from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.amazon.in/Samsung-Control-Fully-Automatic-WW70T502NAN1TL-Inverter/dp/B09KGYCR7Z/ref=sr_1_1_sspa?_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&dib=eyJ2IjoiMSJ9.i598hpMonk2vyjPlFyFQ_URXTCCq9Glj01c89ASnHbgGgTJDga4Ve0dXFrEzKc7KxlpPjBvC8clRxTxC16Po8VLN3NreSL6xEVUR8DYJQDxFa0b4l4I5pcllDJfmENpJQCw4ysHYCI-95f8pETaAZyqCr8tbyfnBVlL0WM8H60M5H3AEelZ_-YIdzz2oWbkrAxQ4v_mZ1baofIlRLZtE-GJ5yds5u2faVGVWpecmpd7saYun0SeVp0_Azful_RVGnRD1R8vk80F82vHX6jLK1m3ltffu9qPVB0y3veqfV1A.0PrrLX5vLVeGMHO2Gi2M1vdMM3gK6akPjn28R7vm4pQ&dib_tag=se&pd_rd_r=73c07f70-bfb6-4ae5-a9f5-ef914d015552&pd_rd_w=xtvUt&pd_rd_wg=BknO6&qid=1765871788&refinements=p_85%3A10440599031&rps=1&s=kitchen&sr=1-1-spons&aref=4GMqVJQVW7&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGZfYnJvd3Nl&th=1'
loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-4-Scout-17B-16E-Instruct'
)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template='Answer the question - {question} from the doc - \n {docs}',
    input_variables=['question', 'docs']
)
parser= StrOutputParser()

chain = prompt | model | parser

output = chain.invoke({'question': 'What is the capacity of this washing machine?','docs': docs[0].page_content})

print(output)