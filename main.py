"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import dvc.api


params = dvc.api.params_show()
chat_params = params['ChatOpenAI']
qa_params = params['Retrieval']

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
llm = ChatOpenAI(temperature=chat_params['temperature'], model_name=chat_params['model_name'], max_retries=chat_params['max_retries'], verbose=chat_params['verbose'])
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever(),
                                                    max_tokens_limit=qa_params['max_tokens_limit'],
                                                    reduce_k_below_max_tokens=qa_params['reduce_k_below_max_tokens'],
                                                    verbose=qa_params['verbose'])

# From here down is all the StreamLit UI.
st.set_page_config(page_title="DVC QA Bot", page_icon=":robot:")
st.header("DVC QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
