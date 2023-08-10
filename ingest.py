"""This is the logic for ingesting Notion data into LangChain."""
from itertools import groupby
import re

from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import json
import dvc.api
import numpy as np
import pandas as pd


JOIN_TOKEN = "\n\n"
DATE_CHARS = 19


def group_consecutive_messages(df):
    # First generate ID to group consecutive messages from same chat
    seq_id = sum(([ii] * len(list(group)) for ii, (_, group) in enumerate(groupby(df['Author'].values.tolist()))), [])
    df['seq'] = seq_id

    # Now the split-apply-combine
    gdf = df.groupby('seq')
    records = []
    for _, df_ in gdf:
        # We might want a different token here
        joined_doc = JOIN_TOKEN.join(df_["Content"].tolist())
        # We could assert all author the same and row 0 is the min date
        record = {"Author": df_["Author"].iloc[0], "Date": df_["Date"].iloc[0], "Content": joined_doc}
        records.append(record)
    df = pd.DataFrame(records)
    return df


def sanitize_name(username):
    username = re.sub('[^a-zA-Z]+', ' ', username)
    username = ' '.join(username.split())
    username = username.title().strip()
    assert len(username) > 0
    return username


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

# Here we load in the data in the format that Notion exports it in.
ps = list(Path("data/discord_csv/").glob("**/*.csv"))

for p in ps:
    # parse package could make this cleaner if we want to
    page_name = str(p).split('__')[1].replace('_', ' ').strip()

    df = pd.read_csv(p, header=0, index_col=False)
    del df['AuthorID']
    del df['Attachments']
    df[df.isnull()] = ""

    # Merge together same chats by same author
    df = group_consecutive_messages(df)

    # Can sanifitize name with `.apply(sanitize_name)` unclear if this is better for the prompts
    # Maybe we should check we didn't create name conflicts, but probably doesn't matter
    df['AuthorClean'] = df['Author']

    # Make formatted message
    df['formatted'] = df.apply(lambda row: f"@{row['AuthorClean']}: {row['Content']}", axis=1)

    # Keep track of chunk sizes
    df['len'] = df['formatted'].str.len() + len(JOIN_TOKEN)
    assert (df['len'] > len(JOIN_TOKEN)).all()
    df['start_position'] = df['len'].cumsum().shift(1, fill_value=0)

    # Check the logic, TODO remove
    assert np.all(np.diff(df['start_position'].values) > 0)  # Should be strict sorted
    joined_doc = JOIN_TOKEN.join(df['formatted'].tolist())
    for ii in range(len(df)):
        assert joined_doc[df['start_position'].iloc[ii]:].startswith(df['formatted'].iloc[ii])

    # Use every message start as potential split
    for ii in range(len(df)):
        start_position = df['start_position'].iloc[ii]
        max_position = start_position + params['chunk_size']
        end_idx = df['start_position'].searchsorted(max_position)
        assert end_idx > ii
        end_idx = end_idx + 5  # Add a few more on the end since we will truncate anyway
        msg = JOIN_TOKEN.join(df['formatted'].iloc[ii:end_idx].tolist())
        assert msg.startswith(df['formatted'].iloc[ii])
        if end_idx < len(df):
            assert len(msg) > params['chunk_size']
        msg = msg[:params['chunk_size']]  # Truncate it back
        msg = msg.rsplit(JOIN_TOKEN, 1)[0].strip()  # Elim partials, but does not distinguish TOKEN inside a message
        assert len(msg) <= params['chunk_size']
        docs.append(msg)

        # Get meta-data from first message
        source = f"discord: {page_name} @{df['Author'].iloc[ii]} [{df['Date'].iloc[ii][:DATE_CHARS]}]"
        metadatas.append({"source": source})

with open("docs.json", "w") as f:
    json.dump(docs, f)

with open("metadatas.json", "w") as f:
    json.dump(metadatas, f)
