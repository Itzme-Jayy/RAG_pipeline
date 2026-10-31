# 📚 RAG Application (Retrieval-Augmented Generation)

An end-to-end Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, **Google Gemini**, and an interactive **Gradio** web interface.

This application enables users to upload custom documents (PDF, DOCX, TXT, Markdown), index them in a persistent vector database, chat with the content through conversational AI with grounded source attribution, and visualize high-dimensional embedding spaces with PCA and t-SNE.

---

## ✨ Features

- **Multi-Format Document Support**: Automatically parse and ingest `.pdf`, `.docx`, `.txt`, and `.md` files.
- **Local Hugging Face Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for dense, fast vector representations without external embedding API costs.
- **Persistent ChromaDB Vector Store**: Indexes chunks into a persistent local collection (`rag_documents`) stored under `./vector_db`.
- **Conversational AI with Gemini**: Integrates with `gemini-3.1-flash-lite` via `ChatGoogleGenerativeAI` to deliver grounded, context-aware answers.
- **Interactive Gradio Web UI**:
  - Two-column chat interface displaying assistant answers alongside retrieved context snippets and sources.
  - Multi-file uploader with permanent disk persistence.
  - One-click embedding generation and ChromaDB indexing.
  - Reset and document deletion controls.
- **Vector Space Visualizations**: Dimensionality reduction using **PCA** and **t-SNE** (`visualize.py`) to inspect vector distributions and cluster quality.

---

## 🏗️ Architecture

```
                                  [ Upload Documents ]
                                           │
                                           ▼
                                 [ Document Loaders ]
                             (PDF, DOCX, TXT, Markdown)
                                           │
                                           ▼
                            [ Recursive Character Splitter ]
                              (size: 500, overlap: 100)
                                           │
                                           ▼
                                [ HuggingFace Embeddings ]
                                  (all-MiniLM-L6-v2)
                                           │
                                           ▼
                             [ ChromaDB Vector Store ]
                                  (./vector_db)
                                     │        │
                   Query Retrieval   │        │   Dimensionality Reduction
                          ┌──────────┘        └──────────┐
                          ▼                              ▼
                 [ Top-K Chunks (k=10) ]         [ PCA & t-SNE ]
                          │                              │
                          ▼                              ▼
               [ Google Gemini LLM ]              [ Scatter Plots ]
            (gemini-3.1-flash-lite)           (pca_visualization.png,
                          │                    tsne_visualization.png)
                          ▼
                 [ Gradio Web UI ]
```

---

## 📁 Project Structure

```text
.
├── app.py                      # Main Gradio application entry point
├── visualize.py                # PCA & t-SNE 2D embedding space visualization script
├── requirements.txt            # Project dependencies
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules for virtualenvs, keys, and DBs
├── implementation/
│   ├── __init__.py
│   ├── answer.py               # Context retrieval and LLM response generation
│   ├── ingest.py               # Document loading, chunking, and ChromaDB indexing
│   ├── upload.py               # File handling and storage in uploaded_documents/
│   └── clear_data.py           # Helper utility to purge uploaded documents
├── uploaded_documents/         # Uploaded raw documents storage
└── vector_db/                  # Persistent ChromaDB sqlite database & vectors
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+ (recommended: Python 3.11 or 3.12)
- A Google Gemini API Key ([Get one from Google AI Studio](https://aistudio.google.com/))

### 2. Clone and Setup Environment

```bash
# Clone the repository (or navigate to project directory)
cd RAG_1st_try

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

*(Optional: Set `HF_TOKEN` if you need higher rate limits for Hugging Face Hub downloads).*

---

## 💻 Running the Application

### 1. Launch the Web Interface

Start the Gradio application:

```bash
python app.py
```

- Gradio will open automatically in your browser (usually at `http://127.0.0.1:7860`).
- **Upload**: Choose your documents (`.pdf`, `.docx`, `.txt`, `.md`) and click **"Upload Documents"**.
- **Vectorize**: Click **"Generate Embeddings and Store in ChromaDB"** to chunk, embed, and store vectors.
- **Chat**: Type questions in the chat box. View both the generated answer and the source chunks side by side.
- **Clear**: Click **"Clear Everything"** or **"delete documents"** to reset files and states.

---

### 2. Visualize Embeddings (PCA & t-SNE)

Once you have ingested documents into ChromaDB, you can visualize how document chunks are distributed across vector space:

```bash
python visualize.py
```

- Generates 2D dimensionality reduction plots.
- Saves high-resolution figures:
  - `pca_visualization.png`
  - `tsne_visualization.png`
- Opens interactive matplotlib plot windows.

> **Note**: Requires at least 3 vectors in the database to run the dimensionality reduction algorithms.

---

### 3. Standalone Script Execution

You can also run ingestion or data cleanup directly via CLI:

```bash
# Ingest all files present in uploaded_documents/ into ChromaDB
python implementation/ingest.py

# Delete all files in uploaded_documents/
python implementation/clear_data.py
```

---

## ⚙️ Configuration Details

| Parameter | Location | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `Embedding Model` | `ingest.py`, `answer.py` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model (384 dimensions) |
| `Chunk Size` | `ingest.py` | `500` characters | Size of each chunk split |
| `Chunk Overlap` | `ingest.py` | `100` characters | Overlapping characters between adjacent chunks |
| `Vector Store` | `ingest.py`, `answer.py` | `./vector_db` | Persistent Chroma database directory |
| `Collection Name` | `ingest.py`, `answer.py` | `rag_documents` | ChromaDB collection name |
| `Retrieval K` | `answer.py` | `10` | Number of context chunks retrieved per query |
| `LLM Model` | `answer.py` | `gemini-3.1-flash-lite` | Google Generative AI model |
| `Temperature` | `answer.py` | `0` | Deterministic outputs for fact-grounded QA |

---

## 🛠️ Troubleshooting

- **Missing `GOOGLE_API_KEY`**:
  Ensure `.env` exists in the root directory and contains `GOOGLE_API_KEY=...`.
- **"At least 3 vectors are needed for this visualization"**:
  Ensure you have ingested documents before executing `python visualize.py`.
- **PDF / DOCX Parsing**:
  `pypdf` and `docx2txt` are included in `requirements.txt`. Ensure all dependencies were installed via `pip install -r requirements.txt`.

