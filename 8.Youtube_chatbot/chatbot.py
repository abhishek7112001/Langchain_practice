from langchain_community.document_loaders import YoutubeLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Indexing(document digestion)
# Document(Transcript) loader
text=""
# try:
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=aircAruvnKk",
    add_video_info=False,
    language=["en"]
)

docs = loader.load()

text =" ".join(i.page_content for i in docs)

print("*************Document Load****************")
print(text)
# except:
#     print("*************Error in doc load****************")
#     print('Transcript of this video is not available')

# Text Splitting(CHunking)

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 50)
chunks= splitter.create_documents([text])
print("*************Text Splitting****************")

print("Total chunks: \n",len(chunks))


# Embedding
embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)
print("*************Vector Embedding****************")
print(vector_store.index_to_docstore_id)



# Retrieval (query + retriever)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})


# Augmentation
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template= """
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the transcript is insufficient, just say you dont know.

        {context}
        Question: {question}
""",
    input_variables=['context', 'question']

)

question = 'From the transcript, is there anything explained about UFO. If you cant find then please return Sorry'
retrieved_docs = retriever.invoke(question)

context_text = " ".join(i.page_content for i in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, 'question': question})
print("*************Prompt****************")

print("Prompt(Question + Context): \n", final_prompt)



# Generation
result = model.invoke(final_prompt)
print("*************OUTPUT****************")
print("Final result: \n", result.content)