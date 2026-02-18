from langchain_text_splitters import RecursiveCharacterTextSplitter
text= '''
    The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding.

    These Llama 4 models mark the beginning of a new era for the Llama ecosystem. We are launching two efficient models in the Llama 4 series, Llama 4 Scout, a 17 billion parameter model with 16 experts, and Llama 4 Maverick, a 17 billion parameter model with 128 experts.
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 200,
    chunk_overlap= 0
)

result= splitter.split_text(text)
print(result)
print(len(result))