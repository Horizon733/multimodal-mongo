import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import unstructured_pytesseract.pytesseract as pt
pt.tesseract_cmd = r"your tesseract path"  # Windows tesseract
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'pdfs/'
figures_directory = 'pdf_figures/'

mongo = MongoClient("mongodb://localhost:27017/?directConnection=true")
collection = mongo["rag"]["mongo_chunks"]

embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection, 
    index_name="pdf_media_idx", 
    text_key="text",
    embedding_key="embedding"
)

# vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="gemma3")
vector_store.create_vector_search_index(
   dimensions = 1024 # The dimensions of the vector embeddings to be indexed
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
custom_rag_prompt = PromptTemplate.from_template(template)
def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
   {"context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | model
   | StrOutputParser()
)


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)

    return "\n\n".join(text_elements)

def extract_text(file_path):
    model_with_image_context = model.bind(images=[file_path])
    response = model_with_image_context.invoke("Tell me what do you see in this picture, describe in short 200 words only")
    print("Extracted text:", response)
    return response

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    print(f"Split into {len(docs)} chunks.")
    print(docs[0])
    return docs

def index_docs(texts):
    vector_store.add_documents(texts)

def retrieve_docs(query):
    return retriever.invoke(query)

def answer_question(question):
    answer = rag_chain.invoke(question)
    return answer

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    text = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_texts = split_text(text)
    index_docs(chunked_texts)

question = st.chat_input()

if question:
    st.chat_message("user").write(question)
    documents = retrieve_docs(question)
    print(f"Retrieved {documents} documents.")
    answer = answer_question(question)
    st.chat_message("assistant").write(answer)
