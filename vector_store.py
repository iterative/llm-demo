import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import json
import dvc.api


params = dvc.api.params_show()['OpenAIEmbeddings']
print(params)

with open("docs.json", "r") as f:
    docs = json.load(f)

with open("metadatas.json", "r") as f:
    metadatas = json.load(f)

print(f"Processing {len(docs)} documents.")

emb = OpenAIEmbeddings(chunk_size=params['chunk_size'],
                       embedding_ctx_length=params['embedding_ctx_length'],
                       max_retries=params['max_retries'],
                       model=params['model'])
emb.show_progress_bar = True

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, emb, metadatas=metadatas)
store.save_local("docs.index")
