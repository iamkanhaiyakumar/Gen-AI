from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
load_dotenv()

# ✅ Hugging Face model
model = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen3-Coder-Next",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=200
)

# Prompt 1
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Explain the following joke:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# ✅ Chain
chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

# Run
print(chain.invoke({"topic": "AI"}))