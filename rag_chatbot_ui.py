from langchain_community.document_loaders import PyPDFLoader #To load the pdf document
from langchain_text_splitters import RecursiveCharacterTextSplitter #Perform chunking
from langchain_huggingface import HuggingFaceEndpoint ,ChatHuggingFace ,HuggingFaceEmbeddings#To initilazie the model
from dotenv import load_dotenv #Load environmental variables
load_dotenv()
import os
from langchain_community.vectorstores import FAISS #Faiss vectorstore
from langchain_core.prompts import PromptTemplate #To create a Instruction prompt
from langchain.retrievers.multi_query import MultiQueryRetriever #Generate multiple query for user's single query
from langchain_groq import ChatGroq #Open source LLM models 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.messages import HumanMessage, AIMessage, trim_messages #Interaction between human and AI
from langchain_core.runnables import RunnableParallel , RunnablePassthrough
import streamlit as st #For UI

st.set_page_config(page_title="Smart Chat Assistant", layout="centered")

# Step-1: Load the PDF Documentfile_path = "Intro_about_AI.pdf"  
file_path = "Intro_about_AI.pdf" #Path to pdf file
loader = PyPDFLoader(file_path) #Load the file
docs= []
for doc in loader.lazy_load():
    docs.append(doc)


## Step-2: Perform Chunking using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "."],  #pripority based seperation
    chunk_size=1000,
    chunk_overlap=120,
    length_function=len,
)

#Apply splitting/chunking
chunks = text_splitter.split_documents(docs)

## Step-3: Load the LLM and Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#load the groq hosted model
model = ChatGroq(model_name = "Llama-3.3-70b-Versatile",max_tokens= 2000)

## Step-4: Creating a VectorStore
VECTOR_STORE_PATH = "faiss_index" # Directory to save the vector store
# Check if vector store exists
if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vector_store.save_local(VECTOR_STORE_PATH)

## Step-5: Creating a prompt template
prompt_template = PromptTemplate(
    template="""You are a Smart Chat Assistant tasked with answering the user question based on the context provided.
Only answer based on the provided context.
If the context does not contain enough information to answer, respond: I do not have enough information to answer your question.
Keep answers concise, clear, and directly relevant to the question.

Context:
{context}
Question:
{input}""",
    input_variables=["context", "input"],
)

## Step-6: Creating a retriever component 
# Multi query retriever takes 2 arguemnts : ("which llm to use","which retriever to use")
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever
    (search_kwargs={"k": 6,'lambda_mult':0.3},  #'k' : number of similari documents to retrieve , 'lambda_mult': to retriever the diverse documents and reduce redundancy
    search_type="mmr"), #Maximum marginal relevance
    llm=model)

## Step-7: Creating a RAG_Chain / Pipeline
parallel_chain = RunnableParallel({
    'context': multiquery_retriever,
    'input': RunnablePassthrough(),
})

rag_chain = parallel_chain | prompt_template | model | StrOutputParser()

## Streamlit App 
st.title("RAG Based Chat-Assistant")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previously generated messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

# Chat input
if user_query := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").markdown(user_query)

    # Only store past 3 conversation due to token limit
    selected_messages = trim_messages(
        st.session_state.messages,
        token_counter=len,
        max_tokens=6,  # <-- allow up to past 6 messages (equivalent to 3 conversations).
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    with st.spinner("Generating response..."):
        response = rag_chain.invoke(selected_messages)

    st.session_state.messages.append(AIMessage(content=response))
    st.chat_message("assistant").markdown(response)
