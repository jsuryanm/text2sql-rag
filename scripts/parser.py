import os 
from unstructured.partition.auto import partition



doc_path = "/home/surya/multidata-rag/data/transformers_paper.pdf"

print(doc_path)

elements = partition(filename=doc_path,
                     strategy='fast')

print(elements)