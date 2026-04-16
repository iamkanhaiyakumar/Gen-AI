from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Your preferred model
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = "https://www.amazon.in/Sony-Headphones-Neck-Band-W128298864-Bluetooth/dp/B07X1TDTQB/?_encoding=UTF8&pd_rd_w=QPdM8&content-id=amzn1.sym.e1111eab-23fc-4988-aab5-fa55f0b13433&pf_rd_p=e1111eab-23fc-4988-aab5-fa55f0b13433&pf_rd_r=Q8KKZ35FQZX6CQZ5W8ZW&pd_rd_wg=nFNxL&pd_rd_r=b403dd98-ddd6-42f9-9d74-c049a0b096ca"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({
    'question': 'What is the product that we are talking about?',
    'text': docs[0].page_content
}))