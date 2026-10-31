
import chromadb
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------
# 1. Connect to existing ChromaDB
# ---------------------------------

DB_PATH = "./vector_db"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(name="rag_documents")


# ---------------------------------
# 2. Extract all embeddings
# ---------------------------------

results = collection.get(
    include=["embeddings", "documents", "metadatas"]
)

embeddings = np.array(results["embeddings"])

documents = results["documents"]

print("Total vectors:", len(embeddings))
print("Embedding dimensions:", embeddings.shape)


if len(embeddings) < 3:
    raise ValueError("At least 3 vectors are needed for this visualization.")


# ---------------------------------
# 3. PCA Visualization
# ---------------------------------

pca = PCA(n_components=2)

pca_vectors = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))

plt.scatter(
    pca_vectors[:, 0],
    pca_vectors[:, 1],
    s=25,
    alpha=0.7
)

plt.title("Vector Database Visualization - PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig("pca_visualization.png", dpi=300)
plt.show()


# ---------------------------------
# 4. t-SNE Visualization
# ---------------------------------

perplexity = min(30, len(embeddings) - 1)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

tsne_vectors = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))

plt.scatter(
    tsne_vectors[:, 0],
    tsne_vectors[:, 1],
    s=25,
    alpha=0.7
)

plt.title("Vector Database Visualization - t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.savefig("tsne_visualization.png", dpi=300)
plt.show()


print("\nVisualization completed!")
print("Saved: pca_visualization.png")
print("Saved: tsne_visualization.png")
