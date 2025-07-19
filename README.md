# RAG Pipeline Overview

A retrieval augmented generation pipeline for an offline LLM to inject further context into prompts, with a simple HTML frontend to interact with the pipeline.

## How To Run

1. Install ollama [here](https://ollama.com/download)
2. In terminal, activate ollama with `ollama` and download the correct LLM with `ollama pull llama3.2`
3. Add the data/documents which are to be used as added context in `/documents` as PDFs.
4. Install dependencies with `pip install -r requirements.txt`
5. Run the app with `python app.py`
6. Follow the link in terminal to the website

## How It Works

- The documents are chunked by a text splitter.
- Each chunk is stored in a list, and a copy of the chunk is vectorized using a sentence transformer, and stored in a vector database.
  - The vector database allows for effective similarity search which is identical to a nearest neighbour search based on the coordinates of the vectors.

- When a prompt is passed into the pipeline, it is converted into a vector by the same sentence transformer model that encoded the data.
- Relevant vectors are retrieved from the database using a nearest neighbour search, and their corresponding indices give the text chunks which were stored in the list.
- The relevant context/chunks are then injected into the final prompt which is passed to the LLM to give a more educated answer.

## Example Responses

### Documents
- [Tipsy cop and tipsy robber: Collisions of biased random walks on graphs](https://www.mdpi.com/2227-7390/9/17/2107)
- [Some remarks on cops and drunk robbers](https://www.sciencedirect.com/science/article/pii/S030439751200789X)
- [A Cop and Drunken Robber Game on n-Dimensional Infinite-Grid Graphs](https://arxiv.org/abs/2208.12829)

Given these documents as context, `dct(G,k)` is the expected capture time of a cops and robbers game on a graph G, given K initial cops. We want to answer our prompt using this context.

### Prompt
When is dct(G, k) < ∞

### Responses

```With RAG
Based on the context provided, we can infer that dct(G, k) is finite if and only if the graph G is a tree. This follows from the fact that in a tree, the cops can always find each other, which allows them to capture the robber in finite time.

To see why this is the case, recall that the capture time is defined as the expected number of rounds until the cops capture the robber. In a tree, the cops can move in any direction, and they will eventually meet each other if they are not already connected by an edge. This means that there exists some k such that the cops can capture the robber in at most k rounds, since they can always find each other.

In contrast, if G is not a tree, then the cops may get lost, and it may take them a long time to find each other. In this case, dct(G, k) may be infinite, since there exists no upper bound on the number of rounds it takes for the cops to capture the robber.

Therefore, we can conclude that dct(G, k) is finite if and only if G is a tree.
```

```Without RAG
DCT (Discrete Cosine Transform) of a discrete-time signal x[n] is defined as:

X[k] = Σx[n] cos(πnk)

where k is the transform index.

The DCT is said to be absolutely convergent if the following condition is met:

|X[k]| < ∞ ∀k ∈ ℤ

In other words, the DCT is absolutely convergent if the absolute value of the k-th transform coefficient does not approach infinity as k approaches positive or negative infinity.

Mathematically, this can be expressed as:

dct(G(k), k) < ∞ ⇔ G(k) → 0 as |k| → ∞

where G(k) is the DCT kernel.
```

## Future Improvement

- The chunking process could be done with an LLM to maximally preserve semantic relationships and avoid early cutoffs.
- Improve website UI
- Create a docker container so the user doesn't have to download ollama and python dependencies themselves
