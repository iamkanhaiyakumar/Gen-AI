from langchain_core.prompts import load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Load saved prompt
template = load_prompt('template.json')

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

# Create chain
chain = template | model

# Run
result = chain.invoke({
    'paper_input': "Attention Is All You Need",
    'style_input': "Beginner-Friendly",
    'length_input': "Short (1-2 paragraphs)"
})

print(result.content)