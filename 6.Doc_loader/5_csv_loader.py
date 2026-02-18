from langchain_community.document_loaders import CSVLoader
loader = CSVLoader('bollywood.csv')

docs = loader.lazy_load()

# print(docs)

for i in docs:
    print(i)