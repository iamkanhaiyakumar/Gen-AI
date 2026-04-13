from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ FIX 1: remove task
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_token
)

# ✅ same structure
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# ✅ FIX 2: add missing prompt
prompt = PromptTemplate(
    template='Write a summary for the following poem:\n{poem}',
    input_variables=['poem']
)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))