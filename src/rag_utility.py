import os
import json

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# loading the embedding model
embedding = HuggingFaceEmbeddings()

# loading the Deepseek-r1 70b model
deepseek_llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0
)

# loading the llama-3 70b model
llama3_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def process_document_to_chromadb(file_name):
    # document directory loader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    # loading the documents
    documents = loader.load()
    # splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=f"{working_dir}/doc_vectorstore")
    return 0


def answer_question(user_question):
    # load the persisted database from disk, and use it as normal.
    vectordb = Chroma(persist_directory=f"{working_dir}/doc_vectorstore",
                      embedding_function=embedding)
    # retriever
    retriever = vectordb.as_retriever()

    # create the chain to answer questions - deepseek-r1
    qa_chain_deepseek = RetrievalQA.from_chain_type(llm=deepseek_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)
    response_deepseek = qa_chain_deepseek.invoke({"query": user_question})
    answer_deepseek = response_deepseek["result"]

    # create the chain to answer questions - llama3
    qa_chain_llama3 = RetrievalQA.from_chain_type(llm=llama3_llm,
                                                    chain_type="stuff",
                                                    retriever=retriever,
                                                    return_source_documents=True)
    response_llama3 = qa_chain_llama3.invoke({"query": user_question})
    answer_llama3 = response_llama3["result"]

    return {"answer_deepseek": answer_deepseek, "answer_llama3": answer_llama3}
