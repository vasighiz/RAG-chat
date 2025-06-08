from document_processor import DocumentProcessor
from embedding_store import EmbeddingStore

# Step 1: Load and split documents
processor = DocumentProcessor(data_dir="../data")
documents = processor.load_documents()
chunks = processor.process_documents(documents)

# Step 2: Embed the chunks
embedder = EmbeddingStore()
embeddings = embedder.create_embeddings(chunks)

# Step 3: Build FAISS index
embedder.create_index(embeddings, chunks)

# Step 4: Run a sample query
sample_query = "What is deep learning?"
top_k = 3
results = embedder.search(sample_query, k=top_k)

print(f"\nTop {top_k} results for query: '{sample_query}'\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"Result {i} (Score: {score:.4f}):\n{doc.page_content[:300]}\n{'-'*40}") 