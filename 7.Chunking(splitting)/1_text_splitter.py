from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(file_path='Computer_Vision.pdf')
docs = loader.load()
# print(docs)

from langchain_text_splitters import CharacterTextSplitter

text= '''
        Please be sure to provide your full legal name, date of birth, and full organization name with all corporate identifiers. Avoid the use of acronyms and special characters. Failure to follow these instructions may prevent you from accessing this model and others on Hugging Face. You will not have the ability to edit this form after submission, so please ensure all information is accurate.
'''

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
    separator=""
)

# splited_text = splitter.split_text(text=docs) # this will return a list
# print(splited_text)
splited_doc = splitter.split_documents(docs)
print(splited_doc[1].page_content)