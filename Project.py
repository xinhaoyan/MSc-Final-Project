import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import json
import base64
import pandas as pd
import tiktoken
import openai
import csv

st.set_page_config(page_title='Msc Project of Xinhao YANG', page_icon=None, layout='centered', initial_sidebar_state='auto')
st.write("Msc Project of Xinhao YANG")


user_api_key = st.sidebar.text_input(
    label="Input your OpenAI API key ",
    placeholder="Paste your openAI API key",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="txt")

MAX_TOKENS = 4096
#Calculate the number of tokens
def count_tokens_with_openai(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

if uploaded_file:
    lines = [line.decode("utf-8").strip() for line in uploaded_file.readlines()]
    df = pd.DataFrame(lines, columns=["text"])


    if not df.empty:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)

        chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=1.0, model_name='gpt-3.5-turbo', openai_api_key=user_api_key),
                                                        retriever=vectors.as_retriever())

        def Chat(query):
            # Calculate the number of tokens for a new issue
            new_question_tokens = count_tokens_with_openai(query)
            # Calculate the number of tokens for historical conversations
            history_tokens = sum([count_tokens_with_openai(q) + count_tokens_with_openai(a) for q, a in st.session_state['history']])
            # Delete the first element of the history (earliest question and answer) if the maximum token limit is exceeded
            while history_tokens + new_question_tokens > MAX_TOKENS:
                removed_question, removed_answer = st.session_state['history'].pop(0)
                history_tokens -= (count_tokens_with_openai(removed_question) + count_tokens_with_openai(removed_answer))

            result = chain({"question": query, "chat_history": st.session_state['history']})
            # Check the number of tokens in a new answer before adding it
            answer_tokens = count_tokens_with_openai(result["answer"])
            if history_tokens + new_question_tokens + answer_tokens <= MAX_TOKENS:
                st.session_state['history'].append((query, result["answer"]))
            else:
                st.warning("The answer is too long to be added to the chat history.")
            return result["answer"]


        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me some question !"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hello！"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
            
            if submit_button and user_input:
                output = Chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

        if st.button('Save Conversation'):
            with open('conversation_history.txt', 'w') as f:
                for item in st.session_state['history']:
                    f.write(str(item) + '\n')
            st.success("Conversation saved successfully!")

        if st.button('Download Conversation'):
            conversation_json = json.dumps(st.session_state['history'], indent=2)
            b64 = base64.b64encode(conversation_json.encode()).decode()
            href = f'<a href="data:text/json;base64,{b64}" download="conversation.json">Download conversation history</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.write("All rows have been processed!")
