# import re
# import subprocess
# import chromadb
# from hazm import Normalizer, sent_tokenize
#
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
# import torch
import logging
# from sentence_transformers import SentenceTransformer
# import os
#
#
# class DocumentQA:
#     def __init__(self, collection_name):
#         self.model = None
#         self.tokenizer = None
#         self.embedder = None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f'device:{self.device}')
#         self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
#         self.collection = self.chroma_client.get_or_create_collection(collection_name)
#
#     def load_model(self):
#         """Load the QA model and tokenizer."""
#         print("Loading QA model...")
#         logging.getLogger("transformers").setLevel(logging.INFO)
#
#         # model_name = "MehdiHosseiniMoghadam/AVA-Mistral-7B-V2"
#         model_name = "lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual"
#         self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#         embedding_model_name = "LABSE"
#         self.embedder = SentenceTransformer(embedding_model_name).to(self.device)
#         print(f'Loaded {embedding_model_name} embedder')
#
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#             self.model.resize_token_embeddings(len(self.tokenizer))
#         print(f'Loaded {model_name} model & tokenizer')
#
#     def pdf_to_text(self, pdf_file):
#         """Extract text from a PDF file using pdftotext."""
#         output_txt = f"{pdf_file}.txt"
#         subprocess.run(["pdftotext", "-layout", pdf_file, output_txt], check=True)
#
#         with open(output_txt, "r", encoding="utf-8") as file:
#             text = file.read()
#
#         return self.clean_text(text)
#
#     def clean_text(self, text):
#         """Remove unwanted characters and normalize."""
#         text = re.sub(r"[\u200E\u200F\u202A-\u202E]", "", text)
#         return re.sub(r"\s+", " ", text).strip()
#
#     def chunk_text(self, text, chunk_size=512):
#         """Split text into chunks."""
#         return sent_tokenize(text)
#
#     def store_pdf_in_db(self, pdf_path):
#         """Extract, chunk, embed, and store a PDF in the vector database."""
#         pdf_name = os.path.basename(pdf_path)
#
#         text = self.pdf_to_text(pdf_path)
#         text_chunks = self.chunk_text(text)
#
#         chunk_embeddings = self.embedder.encode(text_chunks, show_progress_bar=True)
#         for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
#             self.collection.add(
#                 ids=[f"{pdf_name}_{i}"],
#                 embeddings=[embedding.tolist()],
#                 metadatas=[{"text": chunk, "pdf_name": pdf_name}]
#             )
#         print(f"Stored chunks from {pdf_name} in vector database.")
#
#     def find_relevant_chunk(self, question, n_results, selected_pdfs=None):
#         """Perform semantic search and return the best-matching chunk."""
#         question_embedding = self.embedder.encode([question]).tolist()
#         results = self.collection.query(query_embeddings=question_embedding, n_results=n_results)
#
#         relevant_chunks = []
#
#         for i, metadata_list in enumerate(results["metadatas"]):
#             for metadata in metadata_list:
#                 if not selected_pdfs or metadata["pdf_name"] in selected_pdfs:
#                     relevant_chunks.append((metadata["text"], metadata["pdf_name"]))
#
#         return relevant_chunks  # Returns a list of tuples (text, pdf_name)
#
#     def answer_question(self, question, selected_pdfs=None):
#         """Generate an answer using the retrieved chunk."""
#         print(f'selected_pdfs:{selected_pdfs}')
#         relevant_chunks = self.find_relevant_chunk(question,1, selected_pdfs)
#
#         if not relevant_chunks:
#             return "No relevant information found.", None
#
#         # Constructing the prompt with multiple relevant chunks
#         chunks_text = ""
#         for i,chunk in enumerate(relevant_chunks):
#             chunks_text += f"{chunk[0]} "
#
#         # prompt = f'''
#         #             با توجه به شرایط زیر به این سوال پاسخ دهید:
#         #
#         #             {question},
#         #
#         #             متن نوشته: {relevant_chunk}
#         #
#         #             '''
#         # prompt = f"### Human:{prompt}\n### Assistant:"
#         # system_message = """
#         #
#         # لطفاً به سوال زیر پاسخ بده و از اطلاعات بازیابی شده برای استدلال استفاده کن.
#         #
#         # ** دستورالعمل حل سوال:**
#         # 1. ابتدا اطلاعات مرتبط را از متن بازیابی شده استخراج کن.
#         # 2. اگر اطلاعاتی از متن استنباط می‌شود اما به وضوح ذکر نشده، آن را نیز بیان کن.
#         # 3. مراحل استدلال خود را به ترتیب ذکر کن.
#         # 4. در نهایت، نتیجه‌گیری کن و پاسخ نهایی را ارائه بده.
#         #
#         # """
#         system_message = """
#
#      لطفا با توجه به تمامی اطلاعات موجود در متن بازیابی شده زیر، به سوال پاسخ بده. در پاسخ نهایی، سوال را تکرار نکن فقط جواب را به صورت دقیق توضیح بده.
#         """
#
#         user_message = f"""
#         **سوال:** {question}
#
#         **متن بازیابی شده:**
#         {chunks_text}
#
#         """
#
#         prompt = f"""
#         <|im_start|>system
#         {system_message}<|im_end|>
#         <|im_start|>user
#         {user_message}<|im_end|>
#         <|im_start|>assistant
#         """
#
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         generation_config = GenerationConfig(
#             do_sample=True, top_k=40, temperature=0.6, max_new_tokens=600,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#
#         outputs = self.model.generate(**inputs, generation_config=generation_config)
#         answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Extract the Assistant's answer
#         assistant_marker = "<|im_start|>assistant"
#         if assistant_marker in answer:
#             assistant_answer = answer.split(assistant_marker)[1].strip()
#             assistant_answer = assistant_answer.replace("<|im_end|>", "").strip()
#         else:
#             assistant_answer = "No Assistant answer found."
#
#         return assistant_answer, relevant_chunks[0][1]
#
#
# if __name__ == "__main__":
#     print('ok')
#     doc_processor = DocumentQA(collection_name="test6")
#     doc_processor.load_model()
#     # if "test_behzad" in doc_processor.chroma_client.list_collections():
#     #     doc_processor.chroma_client.delete_collection("test_behzad")
#     # print(f'num of records in db:{doc_processor.collection.count()}')
#     # doc_processor.chroma_client.delete_collection("test_behzad")
#     print(f'#### db content: {doc_processor.collection.count()}')
#     # Example Usage: Store multiple PDFs
#     pdf_files = ["./data/source.pdf"]
#     for pdf in pdf_files:
#         doc_processor.store_pdf_in_db(pdf)
#
#     # Example Usage: Query with selected PDFs
#     selected_pdfs = ["source.pdf"]
#     question = "در حال حاظر عمده ترین هدف پدافند غیرعامل چیست؟"
#     answer, source_pdf = doc_processor.answer_question(question, selected_pdfs)
#
#     print(f"Answer: {answer}")
# #     print(f"Source PDF: {source_pdf}")
import requests
import torch
import chromadb
from hazm import sent_tokenize
import re
import subprocess
import os

class DocumentQA:
    def __init__(self, collection_name):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        self.api_url = "http://127.0.0.1:5000"  # FastAPI service URL

    def load_model(self):
        """Load the QA model and tokenizer."""
        print("Loading QA model...")
        logging.getLogger("transformers").setLevel(logging.INFO)

        # model_name = "MehdiHosseiniMoghadam/AVA-Mistral-7B-V2"
        model_name = "lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        print(f'Loaded {model_name} model & tokenizer')

    def pdf_to_text(self, pdf_file):
        """Extract text from a PDF file using pdftotext."""
        output_txt = f"{pdf_file}.txt"
        subprocess.run(["pdftotext", "-layout", pdf_file, output_txt], check=True)

        with open(output_txt, "r", encoding="utf-8") as file:
            text = file.read()

        return self.clean_text(text)

    def clean_text(self, text):
        """Remove unwanted characters and normalize."""
        text = re.sub(r"[\u200E\u200F\u202A-\u202E]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def chunk_text(self, text):
        """Split text into chunks."""
        return sent_tokenize(text)

    def store_pdf_in_db(self, pdf_path):
        """Extract, chunk, embed, and store a PDF in the vector database."""
        pdf_name = os.path.basename(pdf_path)
        text = self.pdf_to_text(pdf_path)
        text_chunks = self.chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            response = requests.post(f"{self.api_url}/embed/", json={"text": chunk})
            embedding = response.json()["embedding"][0]

            self.collection.add(
                ids=[f"{pdf_name}_{i}"],
                embeddings=[embedding],
                metadatas=[{"text": chunk, "pdf_name": pdf_name}]
            )
        print(f"Stored chunks from {pdf_name} in vector database.")

    def find_relevant_chunk(self, question, n_results, selected_pdfs=None):
        """Perform semantic search and return the best-matching chunk."""
        response = requests.post(f"{self.api_url}/embed/", json={"text": question})
        question_embedding = response.json()["embedding"][0]

        results = self.collection.query(query_embeddings=[question_embedding], n_results=n_results)
        relevant_chunks = []

        for metadata_list in results["metadatas"]:
            for metadata in metadata_list:
                if not selected_pdfs or metadata["pdf_name"] in selected_pdfs:
                    relevant_chunks.append((metadata["text"], metadata["pdf_name"]))

        return relevant_chunks

    def answer_question(self, question, selected_pdfs=None):
        """Retrieve the best chunk and generate an answer."""
        relevant_chunks = self.find_relevant_chunk(question, 1, selected_pdfs)

        if not relevant_chunks:
            return "No relevant information found.", None

        context_text = " ".join([chunk[0] for chunk in relevant_chunks])

        system_message = """
        دستور العمل پاسخ دهی
         
        با توجه به اطلاعات موجود در متن بازیابی شده زیر، با فرمت داده شده به سوال پاسخ بده:
        1- اطلاعات موجود در متن بازیابی شده را به عنوان دانش ورودی در نظر بگیر. 
        2- مراحل استدلال را توضیح بده.
        3- جواب نهایی را در جملات کامل و و دقیق ارائه بده.
                """
        user_message = f"""
        **سوال:** {question}

        **متن بازیابی شده:**
        {context_text}
        
        **استدلال:**
        1- ..
        
        **پاسخ نهایی:**
        
        """

        prompt = f"""
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # generation_config = GenerationConfig(
        #     do_sample=True, top_k=40, temperature=0.5, max_new_tokens=1200,
        #     repetition_penalty=1.1,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        # # response = requests.post(f"{self.api_url}/generate/", json={"prompt": user_message})
        # outputs = self.model.generate(**inputs, generation_config=generation_config)
        # answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = requests.post(f"{self.api_url}/generate/", json={
            "prompt": prompt,
            "do_sample": True,
            "top_k": 40,
            "temperature": 0.6,
            "max_new_tokens": 1200
        })

        answer = response.json()["response"]

        assistant_marker = ("<|im_start|>assistant")
        end_think_marker = "</think>"

        thinking_tokens = ""
        output = ""

        if assistant_marker in answer:
            assistant_answer = answer.split(assistant_marker)[1].strip()
            if end_think_marker in assistant_answer:
                thinking_tokens, output = assistant_answer.split(end_think_marker, 1)
                thinking_tokens = thinking_tokens.strip()
                output = output.strip()
            else:
                thinking_tokens = assistant_answer.strip()
                output = "No final output found."
        else:
            output = "No Assistant answer found."
        print(f'answer:{output}')
        print(f'relevant chunk:{relevant_chunks[0][1]}')
        return thinking_tokens[7:], output[15:], relevant_chunks[0][1]
        # return response.json()["response"], relevant_chunks[0][1]

if __name__ == "__main__":
    print('ok')

    doc_processor = DocumentQA(collection_name="test11")
    # doc_processor.load_model()
    # pdf_files = ["./data/source.pdf"]
    # for pdf in pdf_files:
    #     doc_processor.store_pdf_in_db(pdf)

    selected_pdfs = ["source.pdf"]
    question = ' برابر آمار سرشماری سال۱۳۷۸، تعداد شهرها و روستاهای کشور چقدر است؟'
    think, answer, source_pdf = doc_processor.answer_question(question, selected_pdfs)

    print(f"Answer: {answer}")
