"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import json
import dvc.api


params = dvc.api.params_show()['TextSplitter']
print(params)

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("Notion_DB/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=params['chunk_size'],
                                      chunk_overlap=params['chunk_overlap'],
                                      keep_separator=params['keep_separator'],
                                      add_start_index=params['add_start_index'],
                                      separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": str(sources[i])}] * len(splits))

with open("docs.json", "w") as f:
    json.dump(docs, f)

with open("metadatas.json", "w") as f:
    json.dump(metadatas, f)
