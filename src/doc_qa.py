# doc-qa.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import GenerationConfig
from PyPDF2 import PdfReader
import torch
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer


class DocumentQA:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.text_chunks = None
        self.index = None

    # comb models 1 :
    ## model_name = "bert-base-multilingual-cased",embedding_model_name ="all-MiniLM-L6-v2"

    def load_model(self):
        """Load the QA model and tokenizer."""
        print("Loading QA model...")
        logging.getLogger("transformers").setLevel(logging.INFO)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        embedding_model_name ="all-MiniLM-L6-v2" # "LABSE"  #
        self.embedder = SentenceTransformer(embedding_model_name).to(self.device)
        print(f'loaded {embedding_model_name} embedder')

        model_name = "bert-base-multilingual-cased" # "MehdiHosseiniMoghadam/AVA-Mistral-7B-V2"  #
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if the model is generative (e.g., Mistral)
        if "mistral" in model_name.lower():
            self.is_generative = True
            print("Model is generative. Using generation_config for text generation.")
        else:
            self.is_generative = False
            print("Model is extractive. Using standard QA pipeline.")

        # Add a padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize model embeddings
        print(f'loaded {model_name} model & tokenizer')

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def split_text_into_chunks(self, text, chunk_size=512):
        """Split text into smaller chunks for processing."""
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def create_faiss_index(self, chunk_embeddings):
        """Create a FAISS index for efficient similarity search."""
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings))
        return index

    def find_relevant_chunk(self, question):
        """Find the most relevant chunk for a given question."""
        question_embedding = self.embedder.encode([question])
        distances, indices = self.index.search(question_embedding, k=1)
        return self.text_chunks[indices[0][0]]

    def answer_question(self, question):
        """Answer a question based on the document."""
        relevant_chunk = self.find_relevant_chunk(question)
        inputs = self.tokenizer(question, relevant_chunk, return_tensors="pt", truncation=True, padding=True).to(
            self.device)

        # generation_config = GenerationConfig(
        #     do_sample=True,
        #     top_k=1,
        #     temperature=0.99,
        #     max_new_tokens=900,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        outputs = self.model(**inputs) #, generation_config=generation_config
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        return answer

    def initialize_document(self, pdf_file):
        """Initialize the document by extracting text and creating embeddings."""
        text = self.extract_text_from_pdf(pdf_file)
        self.text_chunks = self.split_text_into_chunks(text)
        chunk_embeddings = self.embedder.encode(self.text_chunks)
        self.index = self.create_faiss_index(chunk_embeddings)

# if __name__ == "__main__":
#     print('ok')
#     doc_processor = DocumentQA()
#     doc_processor.load_model()
#     print("loaded model and tokenizer.")
# import faiss
# import numpy as np
# import torch
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# from PyPDF2 import PdfReader
# from sentence_transformers import SentenceTransformer
#
# model_name = "bert-base-multilingual-cased"  # Replace with the actual model name
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text
#
#
# def split_text_into_chunks(text, chunk_size=512):
#     words = text.split()
#     chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
#     return chunks
#
#
# def find_relevant_chunk(question, embedder, index, text_chunks):
#     question_embedding = embedder.encode([question])
#     distances, indices = index.search(question_embedding, k=1)
#     return text_chunks[indices[0][0]]
#
#
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
# pdf_text = extract_text_from_pdf("book.pdf")
# text_chunks = split_text_into_chunks(pdf_text)
#
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
# chunk_embeddings = embedder.encode(text_chunks)
#
# dimension = chunk_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(chunk_embeddings))
#
# while True:
#     question = input("You: ")
#     if question.lower() in ["exit", "quit"]:
#         break
#     answer = answer_question(question, model, tokenizer, text_chunks, embedder, index)
#     print(f"Bot: {answer}")
