from langchain_core.prompts import ChatPromptTemplate

chat_template= ChatPromptTemplate([
    ('system', 'You are an expert in {domain}'),
    ('human', 'Explain in simple terms, what is {topic}?')
])

prompt = chat_template.invoke({'domain' :'football', 'topic' :'goal'})
print(prompt)