from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

text_splitter = SemanticChunker(
    HuggingFaceEmbeddings(), breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)
sample= '''
        Swachhta Abhiyan is a nationwide cleanliness campaign launched in India to promote hygiene, sanitation, and a clean environment. It emphasizes the importance of proper waste management, sanitation facilities, and public participation in maintaining cleanliness in both urban and rural areas. Along with social initiatives, sports like hockey also play a vital role in nation-building. Hockey is considered India’s national sport and has brought immense pride to the country through international achievements. It encourages teamwork, discipline, and physical fitness, inspiring young athletes to represent the nation at global platforms.

        Farhan Akhtar is a versatile Indian artist known for his contributions as an actor, filmmaker, singer, and producer. He made a strong impact with his directorial debut *Dil Chahta Hai*, which redefined modern Hindi cinema. As an actor, he has delivered critically acclaimed performances in films like *Bhaag Milkha Bhaag* and *Zindagi Na Milegi Dobara*. Apart from films, he is actively involved in social causes, including gender equality and youth empowerment, making him an influential figure both on and off the screen.

'''
docs =text_splitter.create_documents([sample])
print(len(docs))
print(docs)