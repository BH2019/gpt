# coding = utf-8
import streamlit as st
from streamlit_chat import message
import requests
import os
from langchain.document_loaders import TextLoader
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.chains.question_answering import load_qa_chain

# url = "https://36kr.com/coop/toutiao/5112440.html?group_id=6509627540695418000&app="
# url = "https://code.visualstudio.com/docs/python/environments"
# url = "https://edition.cnn.com/2023/04/13/business/delta-earnings/index.html"

# res = requests.get(url)
# with open("a.txt", "w") as f:
#    f.write(res.text)

# loader = TextLoader('./a.txt')
loader = TextLoader('./gu.txt', encoding='utf8')

docs = loader.load()

# print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# print (f'Now you have {len(split_docs)} documents')

# OPENAI_API_KEY = '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'
# PINECONE_API_KEY = '9f1a0166-53bd-4c1e-800f-5c02fe1b7f2d'
# PINECONE_API_ENV = 'us-west1-gcp-free'

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_API_ENV"])

index_name = "askgpt"

docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# llm = OpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")
llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo")
chain = load_qa_chain(llm, chain_type="stuff")

st.set_page_config(
   page_title=u"Q&A of Aliens of Extraordinary Ability",
   page_icon=":robot:"
)

st.header(u"Aliens of Extraordinary Ability")
# st.markdown("[Github](https://github.com/ai-yash/st-chat)")

if 'generated' not in st.session_state:
   st.session_state['generated'] = []

if 'past' not in st.session_state:
   st.session_state['past'] = []

def query(q):
   # print("q:", q)
   docs = docsearch.similarity_search(q, 3, include_metadata=True)
   for doc in docs:
      doc.page_content = doc.page_content[:3000]
   generated = chain.run(input_documents=docs, question=q)
   return {'generated': generated}

def get_text():
   input_text = st.text_input("You: ",u"What is Aliens of Extraordinary Ability?", key="input")
   return input_text 


user_input = get_text()

if user_input:
   output = query(user_input)

   st.session_state.past.append(user_input)
   st.session_state.generated.append(output['generated'])

if st.session_state['generated']:

   for i in range(len(st.session_state['generated'])-1, -1, -1):
      message(st.session_state["generated"][i], key=str(i))
      message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
