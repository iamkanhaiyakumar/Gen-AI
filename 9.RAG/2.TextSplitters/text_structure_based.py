from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "Artificial Intelligence (AI) is transforming the way we interact with technology and solve complex problems. From virtual assistants and recommendation systems to self-driving cars and advanced medical diagnostics, AI is becoming an integral part of our daily lives. Machine learning, a subset of AI, enables systems to learn from data and improve their performance over time without being explicitly programmed. With the rise of large language models and generative AI, machines can now generate human-like text, create images, and even assist in software development. However, despite these advancements, challenges such as data privacy, ethical concerns, and model bias continue to be important issues that researchers and developers must address. As AI continues to evolve, it holds the potential to revolutionize industries, enhance productivity, and create new opportunities across the globe." 

splitter =RecursiveCharacterTextSplitter(
    chunk_size=100,
    
    chunk_overlap=3,
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks)