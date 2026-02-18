import streamlit as st

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="wide"
)

st.title("📺 YouTube Transcript Chatbot")
st.write("Ask questions based **only on the YouTube video transcript**.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
with st.sidebar:
    st.header("Configuration")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=aircAruvnKk"
    )

    k_docs = st.slider("Number of retrieved chunks (k)", 1, 10, 4)

    process_btn = st.button("Process Video")


# -------------------------------
# Cache Heavy Operations
# -------------------------------
@st.cache_resource(show_spinner=True)
def build_vectorstore(youtube_url: str):
    # Load transcript
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=False,
        language=["en"]
    )
    docs = loader.load()

    full_text = " ".join(d.page_content for d in docs)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents([full_text])

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# -------------------------------
# LLM Setup (once)
# -------------------------------
@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )
    return ChatHuggingFace(llm=llm)


llm_model = load_llm()

prompt_template = PromptTemplate(
    template="""
You are a helpful assistant.
Answer only from the provided transcript context.
If the transcript is insufficient, just say "Sorry, I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# -------------------------------
# Main Logic
# -------------------------------
if process_btn:
    if not youtube_url:
        st.error("Please provide a YouTube URL.")
    else:
        with st.spinner("Processing YouTube transcript..."):
            vectorstore = build_vectorstore(youtube_url)

        st.success("Video processed successfully!")
        st.session_state["vectorstore"] = vectorstore


# -------------------------------
# Chat Section
# -------------------------------
if "vectorstore" in st.session_state:
    st.subheader("Ask a Question")

    user_question = st.text_input(
        "Your Question",
        placeholder="Is anything explained about UFO?"
    )

    ask_btn = st.button("Ask")

    if ask_btn and user_question:
        retriever = st.session_state["vectorstore"].as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_docs}
        )

        with st.spinner("Retrieving answer..."):
            retrieved_docs = retriever.invoke(user_question)
            context_text = " ".join(d.page_content for d in retrieved_docs)

            final_prompt = prompt_template.invoke({
                "context": context_text,
                "question": user_question
            })

            result = llm_model.invoke(final_prompt)

        st.markdown("### 🤖 Answer")
        st.write(result.content)
