# doc-qa.py

from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer
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

    def temp_chunked_input_data(self):
        return pd.Series(
            data=["آب در دمای صفر درجه یخ میزند"
                , "هدیه مادر برای زهرا یک کفش بود"
                ,  "رنگ مورد علاقه علی آبی است"
                , "ز آنجایی که زهرا دانش آموزبا استعداد و درس خوانی است "
                , "معدل نهایی او در پابه پنجم ابتدایی20بود"
                , "زهرا در یک خانواده پنج نفره همراهمادر و پدر و خواهرش سارا و برادرشعلی زندگی میکند"],

        )

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

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.99,
            max_new_tokens=900,
            pad_token_id=self.tokenizer.eos_token_id
        )
        outputs = self.model(**inputs, generation_config=generation_config)  # ,
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

    def search(self, inp_question, num_res):
        res = {}
        # start_time = time.time()
        question_embedding = self.embedder.encode(inp_question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings)
        # end_time = time.time()
        hits = hits[0]  # Get the hits for the first query

        print("Input question:", inp_question)
        # print("Results (after {:.3f} seconds):".format(end_time - start_time))
        all_input_chunks = self.temp_chunked_input_data()
        for hit in hits[0:num_res]:
            res[hit['corpus_id']] = all_input_chunks[hit['corpus_id']]
        df = pd.DataFrame(list(res.items()), columns=['id', 'res'])
        return df


if __name__ == "__main__":
    print('ok')
    doc_processor = DocumentQA()
    doc_processor.load_model()
    data_chunks = doc_processor.temp_chunked_input_data()

    corpus_embeddings = doc_processor.embedder.encode(data_chunks, show_progress_bar=True, convert_to_tensor=True)

    # doc_processor.create_faiss_index(corpus_embeddings)
    print('ok')
    q = "هدیه مادر برای زهرا چه بود؟"
    res = doc_processor.search(inp_question=q, num_res=1)
    # question_embedding = doc_processor.embedder.encode("زهرا در خانواده چند نفره زندگی میکند ؟", convert_to_tensor=True)
    # hits = util.semantic_search(question_embedding, corpus_embeddings)
    print('ok')
    prompt = f'''

    با توجه به شرایط زیر به این سوال در یک کلمه پاسخ دهید:

    {q},

    متن نوشته: {res['res'][0]} 

    '''
    prompt = f"### Human:{prompt}\n### Assistant:"

    inputs = doc_processor.tokenizer(prompt, return_tensors="pt").to("cuda")

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.99,
        max_new_tokens=900,
        pad_token_id=doc_processor.tokenizer.eos_token_id
    )

    outputs = doc_processor.model.generate(**inputs, generation_config=generation_config)
    print(doc_processor.tokenizer.decode(outputs[0], skip_special_tokens=True))

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
