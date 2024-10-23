import streamlit as st
import os
import pdfplumber
import cohere
from copy import deepcopy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings


def extract_pdf_content(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_content = []
        for page in pdf.pages:
            text = page.extract_text()  
            tables = page.extract_tables()  
            
            page_content = {"text": text, "tables": tables}
            all_content.append(page_content)
        return all_content



def chunk_paragraphs_and_tables(content):
    chunks = []
    
    paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100)

    # For each page, look for paragraphs and tables
    for page in content:
        text = page['text']
        text_chunks = paragraph_splitter.split_text(text)

        # Fixed chunking for paragraphs and add metadata
        chunks.extend([Document(page_content=chunk, metadata = {'chunk_num': text_chunk_num, 'chunk_type' : 'text'}) \
                       for text_chunk_num, chunk in enumerate(text_chunks)])
        
        
        table_chunk_num = len(chunks) 
        for table in page['tables']:
            table_chunks = []
            header = table[0]  
            rows = table[1:]

            # For each table split a certain number of splits manually 
            for i in range(0, len(rows), 2):  
                table_chunk = [header] + rows[i:i+2]
                table_chunks.append(Document(page_content=str(table_chunk), metadata = {'chunk_num': table_chunk_num, 'chunk_type': 'table'}   ))
                table_chunk_num += 1
            chunks.extend(table_chunks)

    return chunks



def get_relevant_content(query, vector_db):
    retrieved_docs = vector_db.similarity_search(query, k=3)
    st.sidebar.text(retrieved_docs)
    content = ''
    for doc in retrieved_docs:
        content += doc.page_content
    return content




def generate (query,generation_model):
    response = generation_model.generate(
        model='command-xlarge-nightly',  
        prompt= query,
        max_tokens=150,  
        temperature=0.7,  
        k=0,  # Top-k sampling (0 disables)
        p=0.9  # Top-p (nucleus) sampling
    )
    yield response.generations[0].text




def rag(prompt, generation_model, vectordb):
    relevant_content = get_relevant_content(prompt, vectordb)
    final_prompt = f'Act as a normal chatbot to answer this: {prompt} ,\
        use this info for help: {relevant_content}'
    yield generate(final_prompt, generation_model)



# --------------------------------- Streamlit ---------------------------------


st.title("RAG Bot with Cohere")

# Sidebar panel for settings
st.sidebar.title("Settings")

# Model selection (just one option for now)
model_choice = st.sidebar.selectbox("Select Model", ["Cohere"])

# Input API keys
embedding_api_key = st.sidebar.text_input("Embedding API Key", type="password")
generation_api_key = st.sidebar.text_input("Generation API Key", type="password")
generation_model = cohere.Client(generation_api_key)  

uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])


if "messages" not in st.session_state:
    st.session_state.messages = []


def messages_to_prompt(messages):
        prompt = ""
        if isinstance(messages,dict):
            prompt += messages['content']
            return prompt 
        
        for message in messages:
            if message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"assistant: {message['content']}\n"
        prompt += "assistant:"  # Adding Bot: at the end to prompt the model to generate the next response
        return prompt

def response_generator(messages, generation_model, vector_db):
    yield rag(messages_to_prompt(messages), generation_model, vector_db)


pdf_content = None
semantic_chunks = None
embedder = None
vector_db = None


if not os.path.exists("streamlit_temp"):
        os.makedirs("streamlit_temp")
if uploaded_file and embedding_api_key:
    new_file_path = os.path.join("streamlit_temp", uploaded_file.name[:-4])
    if not os.path.exists(new_file_path):
        # If any old file exists in the folder, delete it
        for file in os.listdir('streamlit_temp'):
            file_path = os.path.join("streamlit_temp", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        with open(new_file_path, 'w') as new_file:
            new_file.write("")  

        pdf_content = extract_pdf_content(uploaded_file)
        semantic_chunks = chunk_paragraphs_and_tables(pdf_content)
        embedder = CohereEmbeddings(cohere_api_key=embedding_api_key, model='embed-english-v2.0')
        vector_db = FAISS.from_documents(semantic_chunks, embedder)

        # Persist the vector db in the session state to retrieve it when the script refreshes
        st.session_state.vector_db = vector_db      
    
    else:
        embedder = CohereEmbeddings(cohere_api_key=embedding_api_key, model='embed-english-v2.0')
        vector_db = st.session_state.vector_db

if uploaded_file and embedding_api_key and generation_api_key:

    for message in st.session_state.messages:
        print('\n\n\n ------------------ ', message, '----------- \n\n\n')
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = response_generator(st.session_state.messages[-1], generation_model, vector_db)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response })
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            stream = generate( messages_to_prompt(st.session_state.messages[-1])  , generation_model)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
