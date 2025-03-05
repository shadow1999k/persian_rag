import re
import subprocess
import chromadb
from hazm import Normalizer, sent_tokenize

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from sentence_transformers import SentenceTransformer
import os


class DocumentQA:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("doc_chunks")

    def load_model(self):
        """Load the QA model and tokenizer."""
        print("Loading QA model...")
        logging.getLogger("transformers").setLevel(logging.INFO)

        embedding_model_name = "LABSE"
        self.embedder = SentenceTransformer(embedding_model_name).to(self.device)
        print(f'Loaded {embedding_model_name} embedder')

        model_name = "MehdiHosseiniMoghadam/AVA-Mistral-7B-V2"
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

    def chunk_text(self, text, chunk_size=512):
        """Split text into chunks."""
        return sent_tokenize(text)

    def store_pdf_in_db(self, pdf_path):
        """Extract, chunk, embed, and store a PDF in the vector database."""
        pdf_name = os.path.basename(pdf_path)

        text = self.pdf_to_text(pdf_path)
        text_chunks = self.chunk_text(text)

        chunk_embeddings = self.embedder.encode(text_chunks, show_progress_bar=True)
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            self.collection.add(
                ids=[f"{pdf_name}_{i}"],
                embeddings=[embedding.tolist()],
                metadatas=[{"text": chunk, "pdf_name": pdf_name}]
            )
        print(f"Stored chunks from {pdf_name} in vector database.")

    def find_relevant_chunk(self, question, selected_pdfs=None):
        """Perform semantic search and return the best-matching chunk."""
        question_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(query_embeddings=question_embedding, n_results=1)

        for metadata in results["metadatas"][0]:
            if not selected_pdfs or metadata["pdf_name"] in selected_pdfs:
                return metadata["text"], metadata["pdf_name"]

        return None, None

    def answer_question(self, question, selected_pdfs=None):
        """Generate an answer using the retrieved chunk."""
        relevant_chunk, pdf_name = self.find_relevant_chunk(question, selected_pdfs)

        if not relevant_chunk:
            return "No relevant information found.", None

        prompt = f'''
                    با توجه به شرایط زیر به این سوال پاسخ دهید:

                    {question},

                    متن نوشته: {relevant_chunk}

                    '''
        prompt = f"### Human:{prompt}\n### Assistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = GenerationConfig(
            do_sample=True, top_k=1, temperature=0.99, max_new_tokens=900,
            pad_token_id=self.tokenizer.eos_token_id
        )

        outputs = self.model.generate(**inputs, generation_config=generation_config)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_marker = "### Assistant:"

        # Extract the Assistant's answer
        if assistant_marker in answer:
            # Split the text at the Assistant marker and take the part after it
            after_assistant = answer.split(assistant_marker)[1]

            # Find the first occurrence of '#' after the Assistant marker
            end_index = after_assistant.find("#")

            # If '#' is found, extract the text up to that point
            if end_index != -1:
                assistant_answer = after_assistant[:end_index].strip()
            else:
                # If no '#' is found, take the entire text after the Assistant marker
                assistant_answer = after_assistant.strip()
        else:
            assistant_answer = "No Assistant answer found."

        return assistant_answer, pdf_name


if __name__ == "__main__":
    doc_processor = DocumentQA()
    doc_processor.load_model()

    # Example Usage: Store multiple PDFs
    pdf_files = ["./data/book.pdf", "./data/book2.pdf"]
    for pdf in pdf_files:
        doc_processor.store_pdf_in_db(pdf)

    # Example Usage: Query with selected PDFs
    selected_pdfs = ["book2.pdf","book.pdf"]
    question = "آیا علی و سارا تا به حال به آمازون رفته اند؟"
    answer, source_pdf = doc_processor.answer_question(question, selected_pdfs)

    print(f"Answer: {answer}")
    print(f"Source PDF: {source_pdf}")
