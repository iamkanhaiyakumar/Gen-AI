# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
# from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
# from pydantic import BaseModel, Field
# from typing import Literal
# import os

# load_dotenv()

# # ✅ Only changed model to Hugging Face
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=200,
#     temperature=0.3,
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )

# model = ChatHuggingFace(llm=llm)

# parser = StrOutputParser()

# class Feedback(BaseModel):
#     sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# parser2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
#     input_variables=['feedback'],
#     partial_variables={'format_instruction': parser2.get_format_instructions()}
# )

# classifier_chain = prompt1 | model | parser2

# prompt2 = PromptTemplate(
#     template='Write an appropriate response to this positive feedback \n {feedback}',
#     input_variables=['feedback']
# )

# prompt3 = PromptTemplate(
#     template='Write an appropriate response to this negative feedback \n {feedback}',
#     input_variables=['feedback']
# )

# branch_chain = RunnableBranch(
#     (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
#     (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )

# chain = classifier_chain | branch_chain

# print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# # chain.get_graph().print_ascii()


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    max_new_tokens=200,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

# ✅ Take input from terminal
user_input = input("Enter feedback: ")

print(chain.invoke({'feedback': user_input}))