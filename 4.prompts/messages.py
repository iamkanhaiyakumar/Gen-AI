from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

# Invoke model
result = model.invoke(messages)

# Save response
messages.append(AIMessage(content=result.content))

print(messages)