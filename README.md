## RAG system elements : 
 
1- embedding model selection

2-  PDF files reading

3- chunking optimization

4- retrieved chunks formatting

5- prompt engineering

6- questions answering in the target language

7-LLM selection

8- hyperparameters optimization

----------------------------------------------------------------------------
## Key capabilities required for effective RAG (based on ![paper](https://arxiv.org/pdf/2501.04858v1) )
- noise robustness
- negative rejection
- information integration
- counterfactual robustness
------------------------------------------------------------------------------
## Evaluation Datasets: 
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
MCQs generated from this document using GPT-4o assessed the models'
ability to process lengthy, policy-oriented texts with intricate cultural and organizational
references

