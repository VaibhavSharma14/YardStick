import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Set API keys 
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]
os.environ['PINECONE_API_KEY'] = st.secrets["pinecone_api_key"]

# Initialize components
index_name = "rag-index"
embeddings = OpenAIEmbeddings()
llm = OpenAI()

# Streamlit app title
st.markdown("""
    <style>
    .title {
        color: #4CAF50;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
    }
    .subtitle {
        color: #555555;
        text-align: center;
        font-size: 1.5em;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: #AAAAAA;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">RAG-Based QA Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Yardstick</div>', unsafe_allow_html=True)

# Upload document
st.sidebar.header("Upload & Process")
uploaded_file = st.sidebar.file_uploader("Upload a text file", type="txt")

if uploaded_file is not None:
    # Process the document
    file_content = uploaded_file.read().decode("utf-8")
    documents = [Document(page_content=file_content)]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # Add documents to Pinecone
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    vectorstore.add_documents(docs)
    st.sidebar.success("Document uploaded and processed!")

# Main area for query interaction
st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <h2 style="color: #FF5722;">Ask Your Question</h2>
    </div>
""", unsafe_allow_html=True)

query = st.text_input("Enter your query here:", placeholder="Type something interesting...")
if st.button("Get Answer"):
    if uploaded_file is not None:
        results = vectorstore.similarity_search(query, k=1)
        if results:
            result_content = results[0].page_content
            prompt = PromptTemplate.from_template(
                "This is {query} and this is the answer: {results}. Provide the answer in a proper format.\n"
            )
            chain = prompt | llm
            ai_answer = chain.invoke({'query': query, 'results': result_content})
            answer = ai_answer.replace('\n', '').strip()

            st.markdown("""
                <div style="background-color: #E8F5E9; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #388E3C;">Answer</h4>
                    <p style="color: #212121;">{}</p>
                </div>
            """.format(answer), unsafe_allow_html=True)
        else:
            st.error("No relevant answer found.")
    else:
        st.error("Please upload a document first.")

# Footer
st.markdown('<div class="footer">Built with ❤️ by Vaibhav Sharma</div>', unsafe_allow_html=True)