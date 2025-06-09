from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

import dotenv
dotenv.load_dotenv()
FAQ_CHROMA_PATH="chroma_data"
MODEL_NAME="./models/all-mpnet-base-v2"

# First define the system & human prompts -> chat prompt template
roboasistant_system_template_str = """You are a member of the robotic team. 
Your job is to answer questions about the onboarding of entitlements or the product ownership.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.
{context}
"""

roboasistant_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=roboasistant_system_template_str
    )
)

roboasistant_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
roboassistant_messages = [roboasistant_system_prompt, roboasistant_human_prompt]

roboassistant_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=roboassistant_messages
)


# Second define the RAG chain = retriever, chat prompt template, chat model & output parser
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

output_parser = StrOutputParser()

embeddings  = HuggingFaceEmbeddings(
    model_name=MODEL_NAME, #local path
    model_kwargs={'device': 'cpu'}  # or 'cuda' if you have GPU 
)

faqs_vector_db = Chroma(
    persist_directory=FAQ_CHROMA_PATH,
    embedding_function=embeddings,
)

faqs_retriever = faqs_vector_db.as_retriever(k=3)

faqs_roboassistant_chain = (
    {"context": faqs_retriever, "question": RunnablePassthrough()}
    | roboassistant_prompt_template
    | chat_model
    | StrOutputParser()
)

def get_answer(question):
    return faqs_roboassistant_chain.invoke(question)
