from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# HF model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

# Chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

# ✅ Load chat history properly
try:
    with open('chat_history.txt') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
            elif line.startswith("AI:"):
                chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))
except FileNotFoundError:
    pass

print("Loaded History:", chat_history)

# User query
query = "Where is my refund"

# Create chain
chain = chat_template | model

# Invoke model
result = chain.invoke({
    'chat_history': chat_history,
    'query': query
})

# Save new conversation
chat_history.append(HumanMessage(content=query))
chat_history.append(AIMessage(content=result.content))

print("\nAI:", result.content)