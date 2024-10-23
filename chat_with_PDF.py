import pdfplumber
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import sys


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

    
    paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

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
    retrieved_docs = vector_db.similarity_search(query, k=1)
    content = ''
    for doc in retrieved_docs:
        content += doc.page_content
    return content




def generate (query,generation_model):
    response = generation_model.generate(
        model='command-xlarge-nightly',  
        prompt= query,
        max_tokens=100,  
        temperature=0.7,  
        k=0,  # Top-k sampling (0 disables)
        p=0.9  # Top-p (nucleus) sampling
    )
    return response.generations[0].text




def rag(prompt, generation_model, vectordb):
    relevant_content = get_relevant_content(prompt, vectordb)
    final_prompt = f'Act as a normal chatbot to answer this: {prompt} ,\
        use this info for help: {relevant_content},\
        if it is not usefull discard it and not talk about it'
    response = generate(final_prompt, generation_model)
    return response



def main(query):
    # Configurations
    pdf_file_location = "sample.pdf"
    embedding_api_key = 'kQ1bePfavZiNGicrlYIHa71W8M0P0DJbr3Ss89Wt'
    generation_api_key = 'kQ1bePfavZiNGicrlYIHa71W8M0P0DJbr3Ss89Wt'

    # read and chunk the file
    pdf_content = extract_pdf_content(pdf_file_location)
    semantic_chunks = chunk_paragraphs_and_tables(pdf_content)

    # Embedd the chunks, build the vector database and initialize the model
    embedder = CohereEmbeddings(cohere_api_key=embedding_api_key, model='embed-english-v2.0')
    vector_db = FAISS.from_documents(semantic_chunks, embedder)
    generation_model = cohere.Client(generation_api_key)  

    # RAG step using the initialized model and created vector DB
    response = rag(query,generation_model, vector_db)
    print(response)

if __name__ == 'chat_with_PDF.py':
    query = sys.argv[1]
    main(query)


