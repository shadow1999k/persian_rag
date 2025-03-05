# doc-qa.py
import re
import subprocess
import chromadb
from hazm import Normalizer, sent_tokenize

from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from PyPDF2 import PdfReader
import pandas as pd
import torch
import faiss
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, util


class DocumentQA:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.embedder = None
        self.text_chunks = None
        self.index = None
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection("doc_chunks")

    # comb models 1 :
    ## model_name = "bert-base-multilingual-cased",embedding_model_name ="all-MiniLM-L6-v2"

    def load_model(self):
        """Load the QA model and tokenizer."""
        print("Loading QA model...")
        logging.getLogger("transformers").setLevel(logging.INFO)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        embedding_model_name = "LABSE"  # "all-MiniLM-L6-v2" #
        self.embedder = SentenceTransformer(embedding_model_name).to(self.device)
        print(f'loaded {embedding_model_name} embedder')

        model_name = "MehdiHosseiniMoghadam/AVA-Mistral-7B-V2"  # "bert-base-multilingual-cased"  #
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
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

    # def temp_chunked_input_data(self):
    #     return pd.Series(
    #         data=["آب در دمای صفر درجه یخ میزند"
    #             , "هدیه مادر برای زهرا یک کفش بود"
    #             , "رنگ مورد علاقه علی آبی است"
    #             , "ز آنجایی که زهرا دانش آموزبا استعداد و درس خوانی است "
    #             , "معدل نهایی او در پابه پنجم ابتدایی20بود"
    #             , "زهرا در یک خانواده پنج نفره همراهمادر و پدر و خواهرش سارا و برادرشعلی زندگی میکند"],
    #
    #     )

    def pdf_to_pure_text(self, pdf_file, output_txt):
        # Use pdftotext to extract text from the PDF
        subprocess.run(["pdftotext", "-layout", pdf_file, output_txt], check=True)

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file."""
        temp_txt = "temp_output.txt"
        self.pdf_to_pure_text(pdf_file, temp_txt)

        # Step 2: Read the extracted text
        with open(temp_txt, "r", encoding="utf-8") as file:
            extracted_text = file.read()

        # Step 3: Clean the text
        cleaned_text = self.clean_text(extracted_text)

        # Step 4: Normalize the text
        normalized_text = self.normalize_text(cleaned_text)

        # Step 5: Chunk the text into sentences
        sentences = self.chunk_text_into_sentences(normalized_text)
        return sentences

    def clean_text(self, text):
        # Remove Unicode control characters (RLE, LRE, PDF, etc.)
        text = re.sub(r"[\u200E\u200F\u202A-\u202E]", "", text)
        # Remove extra spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_text(self, text):
        # Normalize Persian text using hazm
        normalizer = Normalizer()
        return normalizer.normalize(text)

    def chunk_text_into_sentences(self, text):
        # Split the text into sentences using hazm
        return sent_tokenize(text)

    # def split_text_into_chunks(self, text, chunk_size=512):
    #     """Split text into smaller chunks for processing."""
    #     words = text.split()
    #     chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    #     return chunks

    # def create_faiss_index(self, chunk_embeddings):
    #     """Create a FAISS index for efficient similarity search."""
    #     dimension = chunk_embeddings.shape[1]
    #     index = faiss.IndexFlatL2(dimension)
    #     index.add(np.array(chunk_embeddings))
    #     return index
    def store_chunks_in_db(self, text_chunks):
        """Store document chunks in ChromaDB."""
        chunk_embeddings = self.embedder.encode(text_chunks, show_progress_bar=True)
        for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedding.tolist()],
                metadatas=[{"text": chunk}]
            )
        print("Text chunks stored in vector database.")

    def find_relevant_chunk(self, question):
        """Find the most relevant chunk for a given question."""
        question_embedding = self.embedder.encode([question])
        results = self.collection.query(query_embeddings=question_embedding, n_results=1)
        return results["metadatas"][0][0]["text"]

    def answer_question(self, question):
        """Answer a question based on the document."""
        res = self.find_relevant_chunk(question)
        # question_embedding = doc_processor.embedder.encode("زهرا در خانواده چند نفره زندگی میکند ؟", convert_to_tensor=True)
        # hits = util.semantic_search(question_embedding, corpus_embeddings)

        prompt = f'''

            با توجه به شرایط زیر به این سوال پاسخ دهید:

            {q},

            متن نوشته: {res}

            '''
        prompt = f"### Human:{prompt}\n### Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.99,
            max_new_tokens=900,
            pad_token_id=doc_processor.tokenizer.eos_token_id
        )

        outputs = self.model.generate(**inputs, generation_config=generation_config)

        # Decode the generated text
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the Assistant's answer
        assistant_marker = "### Assistant:"

        # Extract the Assistant's answer
        if assistant_marker in full_output:
            # Split the text at the Assistant marker and take the part after it
            after_assistant = full_output.split(assistant_marker)[1]

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
        # Print the Assistant's answer
        print(f'answer: {assistant_answer}')
        return assistant_answer
        # relevant_chunk = self.find_relevant_chunk(question)
        # inputs = self.tokenizer(question, relevant_chunk, return_tensors="pt", truncation=True, padding=True).to(
        #     self.device)
        #
        # generation_config = GenerationConfig(
        #     do_sample=True,
        #     top_k=1,
        #     temperature=0.99,
        #     max_new_tokens=900,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )
        # outputs = self.model(**inputs, generation_config=generation_config)  # ,
        # answer_start = torch.argmax(outputs.start_logits)
        # answer_end = torch.argmax(outputs.end_logits) + 1
        # answer = self.tokenizer.convert_tokens_to_string(
        #     self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    def initialize_document(self, pdf_file):
        """Initialize the document by extracting text and creating embeddings."""
        self.text_chunks = self.extract_text_from_pdf(pdf_file)
        self.store_chunks_in_db(self.text_chunks)
        # chunk_embeddings = self.embedder.encode(self.text_chunks)
        # self.index = self.create_faiss_index(chunk_embeddings)

    # def search(self, inp_question, num_res):
    #     res = {}
    #     # start_time = time.time()
    #     question_embedding = self.embedder.encode(inp_question, convert_to_tensor=True)
    #     hits = util.semantic_search(question_embedding, corpus_embeddings)
    #     hits = hits[0]  # Get the hits for the first query
    #     for hit in hits[0:num_res]:
    #         res[hit['corpus_id']] = data_chunks[hit['corpus_id']]
    #     df = pd.DataFrame(list(res.items()), columns=['id', 'res'])
    #     print(f'search --> done. chunk:{df}')
    #     return df


if __name__ == "__main__":
    print('ok')
    doc_processor = DocumentQA()
    doc_processor.load_model()

    doc_processor.initialize_document('./data/book.pdf')
    # data_chunks = doc_processor.extract_text_from_pdf('./data/book.pdf')
    # corpus_embeddings = doc_processor.embedder.encode(data_chunks, show_progress_bar=True, convert_to_tensor=True)
    # print(f'answer: corpus embedding --> done')
    # doc_processor.create_faiss_index(corpus_embeddings)
    q = "رنگ مورد علاقه علی چیست؟"
    print(f'qestion:{q}')
    rag_answer = doc_processor.answer_question(q)
    print(f'answer: {rag_answer}')

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
