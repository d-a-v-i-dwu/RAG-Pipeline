import ollama
from vector_store import retrieve_similar_chunks

client = ollama.Client()
ollama_model = "llama3.2"

def get_response(prompt):
    # Given a prompt, retrieve relevant chunks from documents and inject it
    context = retrieve_similar_chunks(prompt)
    final_prompt = f"Using this context: {context}\n Answer this prompt: {prompt}"

    response = client.generate(model=ollama_model, prompt=prompt)
    return(response.response)