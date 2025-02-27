# Table of contents
0. [paper 1](https://arxiv.org/pdf/2501.04858v1)
1. [RAG system elements](#RAG_system_elements)
2. [Key capabilities required for effective RAG](#second)
3. [Evaluation Datasets](#third)
4. [Embedding Model Evaluation](#fourth)
5. [Best LLMs (as baseline)](#fifth)

## RAG system elements :  <a name="RAG_system_elements"></a>
 
1- embedding model selection

2-  PDF files reading

3- chunking optimization

4- retrieved chunks formatting

5- prompt engineering

6- questions answering in the target language

7-LLM selection

8- hyperparameters optimization

----------------------------------------------------------------------------
## Key capabilities required for effective RAG (based on ![paper](https://arxiv.org/pdf/2501.04858v1) ) <a name="second"></a>
- noise robustness
- negative rejection
- information integration
- counterfactual robustness
------------------------------------------------------------------------------
## Evaluation Datasets: <a name="third"></a>
1- General Knowledge Dataset 

PQuad, a Persian-language reading
comprehension dataset sourced from Wikipedia articles. Contains approximately 80,000 questions, 25% of which are classified as unanswerable. It is divided into training (63,994), validation (7,976), and test (8,002) subset.

 The wide range of topics covered by
Wikipedia makes this dataset particularly suitable for evaluating the ''generalization ability'' of
RAG models in retrieving and generating information from diverse, open-domain content

2- Scientific-Specialized Dataset

The scientific-specialized dataset was created using content from the Persian-language textbook
General Physical Education.  This textbook contains comprehensive information on the
philosophy of physical education, exercise physiology, and health sciences, making it an ideal
source for evaluating models' performance in handling "domain-specific language".

3- Organizational Report Dataset

The third dataset was built using the Fundamental Transformation Document of Education in
Iran, a formal policy document detailing key strategies and frameworks for overhauling the
Iranian education system by the year 1404 (2025). This document contains structured, formal
language, along with socio-political and cultural terminology, making it an important test case for
evaluating how well models can retrieve and generate information from highly formalized and
context-specific texts.
MCQs (Multiple-Choice Question) generated from this document using GPT-4o assessed the models'
ability to process lengthy, policy-oriented texts with intricate cultural and organizational
references

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |


## Embedding Model Evaluation <a name="fourth"></a>
- MatinaSRoberta
- LaBSE[17]
- L12-V2
- Qwen2-7[18]
- Alibaba/gte

The models chosen are compatible with the Sentence-Transformers framework, ensuring support for creating high-quality sentence embeddings.


## Best LLMs (as baseline) <a name="fifth"></a>
The capabilities of the LLMs were evaluated using a Multiple-Choice Question Answering
(MCQA), built on the LlamaIndex framework, which was applied to each of the three datasets
(mentioned above)
- LLaMA 3.1 (8B & 70B)
- Qwen 2 (7B & 72B)
- Gemma 1.1, and Gemma 2
