from langchain_core.prompts import PromptTemplate
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

# template
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# create chain (IMPORTANT change)
chain = template2 | model

# invoke
result = chain.invoke({'name': 'kanhaiya'})

print(result.content)