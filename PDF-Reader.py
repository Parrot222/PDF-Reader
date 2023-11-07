from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from google_speech import Speech
from PyPDF2 import PdfReader
import streamlit as st
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0, streaming=True, openai_api_key=api_key)



#分析多個PDF檔案內文字
def pdf2text(docs):
    text = ''
    for pdf in docs:
        pdf_file = PdfReader(pdf)
        for page in pdf_file.pages:
            text += page.extract_text()
            
    #處理中文字編碼
    text = text.encode('utf8')
    text = str(text, 'utf8')
    return text

#因為輸入語言模型自數有限所以須進行文字分割
def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap  = 100, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(text)
    return texts

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

#判斷輸出語音使用語言
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

#建立回答問題模型
from langchain.chains.question_answering import load_qa_chain

def QandA(document, query):
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    docs = document.similarity_search(query)
    anw = chain.run(input_documents=docs, question=query)
    return anw

def main():
    #Question
    query = st.chat_input("Ask questions about your PDF file:")
    have_file = False

    if query:
        if 'vectorstore' not in st.session_state:
            st.warning('Files not found.', icon="⚠️")
        else:
            vectorstore = st.session_state.vectorstore
            docs = vectorstore.similarity_search(query=query, k=1)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            st.chat_message("USER").write(query)
            with st.spinner("思考中..."):
                response = chain.run(input_documents=docs, question=query)
                st.chat_message("AI").write(response)
                if(is_Chinese(response)):
                    speech = Speech(response, "cmn-TW")
                else:
                    speech = Speech(response, "en")
                speech.save("output.mp3")
                st.audio("output.mp3")

    with st.sidebar:
        st.subheader("Your PDF files")
        pdf_docs = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

        if(st.button("Run")):
            with st.spinner("Proccessing"):
                # Get PDF text
                raw_text = pdf2text(pdf_docs)

                # Get that text chunks
                text_chunks = get_text_chunk(raw_text)

                # Get Vectorstore
                st.session_state.vectorstore = get_vectorstore(text_chunks)

                have_file = True
                
        if(have_file):
            st.write("Done!")

if __name__ == '__main__':
    main()
