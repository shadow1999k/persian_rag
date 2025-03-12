import streamlit as st
from doc_qa import DocumentQA  # Assuming your backend is in a file named backend.py

# Initialize DocumentQA instance
doc_processor = DocumentQA(collection_name="test11")

st.title("📄 Document-based QA System")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        with open(f"./data/{file.name}", "wb") as f:
            f.write(file.getbuffer())
        doc_processor.store_pdf_in_db(f"./data/{file.name}")
        st.success(f"✅ {file.name} ready to use!")

# Question input
question = st.text_area("💬 Ask a question:", height=150)
# Button to submit question
if st.button("Get Answer"):
    if question.strip():
        think, answer, source_pdf = doc_processor.answer_question(question)

        # Display answer first
        st.write("### ✅ Answer:")
        st.success(answer)

        # Display reasoning (استدلال)
        st.write("### 🧠 استدلال:")
        st.info(think)

    else:
        st.error("Please enter a question before clicking 'Get Answer'.")

# import streamlit as st
# import os
# from doc_qa import DocumentQA
#
# # Initialize the RAG system and load models only ONCE
# st.title("📖 Persian Document QA")
# st.write("Upload PDF files, select which ones to search, and ask your question.")
#
# # Load the model **once** before file uploads
# @st.cache_resource
# def initialize_rag_system():
#     doc_processor = DocumentQA()
#     doc_processor.load_model()  # Load models only once
#     return doc_processor
#
# doc_processor = initialize_rag_system()
#
# # Directory to store PDFs
# PDF_DIR = "./data"
# os.makedirs(PDF_DIR, exist_ok=True)
#
# # Session state to track uploaded files
# if "uploaded_pdfs" not in st.session_state:
#     st.session_state.uploaded_pdfs = []
#
# # File upload section
# uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)
#
# if st.button("Index PDFs"):
#     if uploaded_files:
#         for file in uploaded_files:
#             file_path = os.path.join(PDF_DIR, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())  # Save uploaded file
#
#             # Only index if it's a new upload
#             if file.name not in st.session_state.uploaded_pdfs:
#                 doc_processor.store_pdf_in_db(file_path)
#                 st.session_state.uploaded_pdfs.append(file.name)  # Track uploaded PDFs
#
#         st.success("✅ PDFs indexed successfully!")
#     else:
#         st.warning("⚠️ Please upload at least one PDF.")
#
# # Show checkboxes for selecting PDFs (only after files are uploaded)
# if st.session_state.uploaded_pdfs:
#     st.write("📑 **Select PDFs to search**")
#     selected_pdfs = []
#     for pdf in st.session_state.uploaded_pdfs:
#         if st.checkbox(pdf, value=True, key=pdf):  # Default: selected
#             selected_pdfs.append(pdf)
#
#     # Question input and search
#     question = st.text_input("❓ Enter your question:")
#     if st.button("Search & Answer"):
#         if not question:
#             st.warning("⚠️ Please enter a question!")
#         elif not selected_pdfs:
#             st.warning("⚠️ Please select at least one PDF to search.")
#         else:
#             answer, source_pdf = doc_processor.answer_question(question, selected_pdfs)
#             st.write(f"**💡 Answer:** {answer}")
#             st.write(f"📄 **Source PDF:** {source_pdf if source_pdf else 'No relevant document found'}")
