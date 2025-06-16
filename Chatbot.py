import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Set environment variables from Streamlit secrets
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")


# Setup Streamlit app
st.title("Conversational RAG With PDF With Chat History")
st.write("Upload PDF And Ask Questions")

# Import the Groq API
from langchain_groq import ChatGroq
llm= ChatGroq(groq_api_key=groq_api_key,model="gemma2-9b-it")


# Input session_id (which session from user) -->string form(eg- "user123")
session_id=st.text_input("Session ID",value="default_session") 

#st.session_state = {
   # 'chat_history': ["hi", "how are you?"],
  #  'store': <FAISS vectorstore>,
 #   'user_name': "Mayank",
 # 'session_id':default_session
#}

# Create 'Store' dictonary inside session_state dictonary which will contain session_id and their chat
if 'store' not in st.session_state:
   st.session_state.store={}

   
# Upload File
upload_file=st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

if upload_file:

    # Save to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
     tmp_file.write(upload_file.read())
     tmp_file_path = tmp_file.name

   #then Load the Uploaded file
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    # DATA->DATA CHUNKS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_split=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    docs_split=text_split.split_documents(docs)

    # TEXT->VECTORS
    from langchain.embeddings import HuggingFaceEmbeddings
    # Create embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'},encode_kwargs={"normalize_embeddings": True})  # Force it to CPU)  # or any other supported model

    # Store to VectorDB
    from langchain.vectorstores import Chroma
    vectorstore=Chroma.from_documents(documents=docs_split,embedding=embeddings)

    # Create Retriever
    from langchain_core import retrievers
    retriever=vectorstore.as_retriever()
    
    #Create History Prompt (NO CONTEXT)-->used to reframe the ques. for the reteriever with help of chat_history
    from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
    history_prompt=ChatPromptTemplate.from_messages([
        ("system","Given a Chat History and latest User Question, just reformulate the User question again if needed , return as it is otherwise"),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    prompt=ChatPromptTemplate.from_messages([
        #Set rules and behaviour tone for LLM
        ("system","You are helpful assistant so answer the user query while remebering previous context"),
        MessagesPlaceholder("chat_history"),
        # Current user query and the Context(tells it what to answer right now)
        ("human","Answer the following question using the provided context:\n\n<context>\n{context}\n</context>\n\nQuestion: {input}")
    ])
    # Create history_aware_retriever (reframe the user ques. with given chat_history with help of llm)
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    history_retiever=create_history_aware_retriever(llm,retriever,history_prompt)

    # Create Document Chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    document_chain=create_stuff_documents_chain(llm,prompt)
    
    # Create Retiever Chain( connect retriever with -->prompt-->llm)
    from langchain.chains.retrieval import create_retrieval_chain
    retriever_chain=create_retrieval_chain(history_retiever,document_chain)

    # ✅ Session-aware history manager
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory

    
    #-->it takes a session string (like "default_session" or "user123")
    #It will return a chat message history object for that session.
    def get_session_history(session:str)->BaseChatMessageHistory:  
# if that session is not present inside 'store' then create that empty 'session' which will contain its chat history
         if session not in st.session_state.store:            
             st.session_state.store[session]=StreamlitChatMessageHistory()

         return st.session_state.store[session]  #--> return chat message history of that 'session'
    

   # works as manager that manages everything
    conversational_rag_chain=RunnableWithMessageHistory(
        #Appends both the user question & LLM response to chat history — keeping memory updated.

            retriever_chain,  # call your actual RAG chain (retriever + prompt + LLM)

            # Calls your get_session_history() function to get chat history for that session.
            get_session_history, 

            #Passes the user question to the chain 
            input_messages_key="input", # input dict key that holds the user's question

            #Adds the chat history into the chain input using
            history_messages_key="chat_history", # what name the chain expects for history

            # where to store the LLM response(Where in the output dictionary should I find the final message (i.e., the LLM’s response) that needs to be stored in chat history?"-->rensponse["answer"])
            output_messages_key="answer"  
        )

# This should be outside the `if upload_file` block, so it works after upload    
user_input=st.text_input("Ask Your Question")

if user_input and upload_file:
        #get session chat_history,if not there then create one empty dictonary for chat_history
        session_history=get_session_history(session_id) 
        response=conversational_rag_chain.invoke(
           {"input":user_input},
           config={
             "configurable":{"session_id":session_id}  #config={"configurable": {"session_id": "user123"}}
           }
        )
        
        st.write("Response:",response["answer"])
        st.write(st.session_state.store)
        st.write("Chat History:", session_history.messages)

#---------------------------------------------------------------------
#Response structure:
#{
#  "answer": "This is the final response generated by the LLM.",
#  "context": [
#      Document(page_content="...relevant chunk 1...", metadata={...}),
#      Document(page_content="...relevant chunk 2...", metadata={...}),
#      ...
#  ]
#}