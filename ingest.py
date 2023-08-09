"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import json
import dvc.api
import pandas as pd
from parse import search


JOIN_TOKEN = "\n\n"
METADATA_TOKEN = "\t"
DATE_CHARS = 19

params = dvc.api.params_show()['TextSplitter']
print(params)

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("data/content/").glob("**/*.md"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(f"dvc.org: {str(p)}")

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("data/discord_csv/").glob("**/*.csv"))

for p in ps:
    df = pd.read_csv(p, header=0, index_col=False)
    del df['AuthorID']
    del df['Attachments']
    df[df.isnull()] = ""
    assert not df['Author'].str.contains(METADATA_TOKEN).any()
    assert not df['Date'].str.contains(METADATA_TOKEN).any()
    assert not df['Content'].str.contains(METADATA_TOKEN).any()
    # We could try eliminating instances of \n\n from content as well
    df['formatted'] = df.apply(lambda row: f"@{row['Author']} [{METADATA_TOKEN}{row['Date'][:DATE_CHARS]}{METADATA_TOKEN}]: {row['Content']}", axis=1)
    joined_doc = '\n\n'.join(df['formatted'].tolist())
    data.append(joined_doc)

    # parse package could make this cleaner if we want to
    page_name = str(p).split('__')[1].replace('_', ' ').strip()
    sources.append(f"discord: {page_name}")

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
    for j, ss in enumerate(splits):
        parsed_meta = search(f"@{{}} [{METADATA_TOKEN}{{}}{METADATA_TOKEN}]: ", ss)

        if parsed_meta is None:
            source = f"{sources[i]} {int(100 * (j / len(splits)))}%"
        else:
            author, date = parsed_meta
            source = f"{sources[i]} @{author} [{date}]"

        ss = ss.replace(METADATA_TOKEN, "")
        docs.append(ss)
        metadatas.append({"source": source})

with open("docs.json", "w") as f:
    json.dump(docs, f)

with open("metadatas.json", "w") as f:
    json.dump(metadatas, f)
