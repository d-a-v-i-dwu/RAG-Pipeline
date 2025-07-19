import os
import faiss
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# The model we're using to convert text into vectors to preserve semantic meaning
model = SentenceTransformer("all-MiniLM-L6-v2")
# A smart splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

# The vector database, uses L2 Euclidean distance to find nearest neighbours, maps in 384 dimensions which matches the model input size
index = faiss.IndexFlatL2(384)

# Each element is a chunk of text
all_chunks = []

# Function to convert pdf into text
def extract_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to create the vector database
def create_vector_store():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents = os.path.join(current_dir, "documents")

    # For each PDF
    for filename in os.listdir(documents):
        if filename.lower().endswith(".pdf"):
            # Extract the text from the PDF
            pdf_path = os.path.join(documents, filename)
            text = extract_pdf_text(pdf_path)

            # Split the text into chunks and then append to all chunks. We need this
            # so that after finding the index of relevant vector, we can use that index
            # to get the original text chunk to use as context for our prompt
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)

            # Convert the chunks into vectors then add to vector database
            embeddings = model.encode(chunks).astype("float32")
            index.add(embeddings)

create_vector_store()

def retrieve_similar_chunks(prompt):
    # Encode the prompt as a vector, then do nearest neighbour search for similar vectors,
    # since the encoding preserves semantic relationships
    prompt_embedding = model.encode([prompt]).astype("float32")
    D, I = index.search(prompt_embedding, k=3)
    
    # I contains indices of relevant vectors in groups of k=3, so I[0] are the 3 most 
    # relevant vectors
    return "\n\n".join([all_chunks[i] for i in I[0]])
