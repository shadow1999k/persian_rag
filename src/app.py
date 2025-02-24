
# app.py
import streamlit as st
from doc_qa import DocumentQA

# Initialize session state
if "qa_system" not in st.session_state:
    st.session_state.qa_system = DocumentQA()
    st.session_state.qa_system.load_model()
    st.session_state.chat_history = []
    st.session_state.document_initialized = False

# Streamlit UI
st.title("Document Q/A System")
st.write("Upload a PDF and ask questions about its content.")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
if pdf_file is not None and not st.session_state.document_initialized:
    st.session_state.qa_system.initialize_document(pdf_file)
    st.session_state.document_initialized = True
    st.success("PDF uploaded and processed successfully!")

# Chat interface
if st.session_state.document_initialized:
    st.write("### Chat with the Document")
    user_input = st.text_input("Ask a question:")
    if user_input:
        # Get the answer from the backend
        answer = st.session_state.qa_system.answer_question(user_input)

        # Update chat history
        st.session_state.chat_history.append({"question": user_input, "answer": answer})

        # Display chat history
        st.write("### Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Bot:** {chat['answer']}")
            st.write("---")
else:
    st.warning("Please upload a PDF file to start.")
###########################################################
# import streamlit as st
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# from PyPDF2 import PdfReader
# import torch
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
#
# # Load the model and tokenizer
# @st.cache_resource
# def load_model():
#     model_name = "bert-base-multilingual-cased"
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
#     return model, tokenizer, embedder
#
#
# # Extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text
#
#
# # Split text into chunks
# def split_text_into_chunks(text, chunk_size=512):
#     words = text.split()
#     chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
#     return chunks
#
#
# # Create FAISS index
# def create_faiss_index(chunk_embeddings):
#     dimension = chunk_embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(chunk_embeddings))
#     return index
#
#
# # Find relevant chunk
# def find_relevant_chunk(question, embedder, index, text_chunks):
#     question_embedding = embedder.encode([question])
#     distances, indices = index.search(question_embedding, k=1)
#     return text_chunks[indices[0][0]]
#
#
# # Answer question
# def answer_question(question, model, tokenizer, text_chunks, embedder, index):
#     relevant_chunk = find_relevant_chunk(question, embedder, index, text_chunks)
#     inputs = tokenizer(question, relevant_chunk, return_tensors="pt", truncation=True, padding=True)
#     outputs = model(**inputs)
#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(
#         tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
#     return answer
#
#
# # Streamlit UI
# def main():
#     st.title("Ask your doc :)")
#
#     # Upload PDF
#     pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
#     if pdf_file is not None:
#         st.write("PDF uploaded successfully!")
#         text = extract_text_from_pdf(pdf_file)
#         text_chunks = split_text_into_chunks(text)
#
#         # Load model and embedder
#         model, tokenizer, embedder = load_model()
#
#         # Create embeddings and FAISS index
#         chunk_embeddings = embedder.encode(text_chunks)
#         index = create_faiss_index(chunk_embeddings)
#
#         # Question input
#         question = st.text_input("Ask a question about the document:")
#         if question:
#             answer = answer_question(question, model, tokenizer, text_chunks, embedder, index)
#             st.write(f"**Answer:** {answer}")
#
#
# if __name__ == "__main__":
#     main()