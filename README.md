# Medical RAG QA System

A Retrieval-Augmented Generation (RAG) system for medical question answering using LangChain, Ollama, and FAISS.

## 📋 Overview

This system provides accurate, context-based answers to medical questions by retrieving relevant information from a database of medical transcriptions and generating responses using a local Large Language Model (LLM).

### Key Features

- **Local Deployment**: Runs entirely on your machine with Ollama (no API keys needed)
- **Medical Knowledge Base**: Built on 4,999 medical transcription records
- **Source Attribution**: Provides citations with medical specialties and keywords
- **Interactive Web UI**: Streamlit-based interface for easy interaction
- **Comprehensive Evaluation**: Tested on 33+ medical queries across 10+ specialties

## 🏗️ Architecture

```
User Query → Embedding → FAISS Vector Search → Top-K Retrieval → LLM Generation → Answer + Sources
```

**Components:**
- **Dataset**: Medical Transcriptions from Kaggle
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for similarity search
- **LLM**: Llama 3.2 via Ollama
- **Framework**: LangChain

## 🚀 Quick Start

### Prerequisites

1. **Python 3.12+** (Python 3.14 recommended)
2. **Ollama** - Install from [ollama.com](https://ollama.com)
3. **Dataset** - Download from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

### Installation

1. **Clone or download this project**
```bash
cd GenAI_RAG/Task1
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and start Ollama**
```bash
# Download Ollama from https://ollama.com
# Then pull the Llama 3.2 model
ollama pull llama3.2
```

4. **Download the dataset**
- Download Medical Transcriptions from Kaggle
- Extract `mtsamples.csv` to the `data/` folder

### Running the Application

#### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

The web interface will open at `http://localhost:8501`

#### Option 2: Jupyter Notebook

```bash
jupyter notebook Medical_RAG_QA_System.ipynb
```

Run all cells sequentially to build the RAG system and evaluate it.

## 📁 Project Structure

```
Task1/
├── Medical_RAG_QA_System.ipynb    # Main development notebook
├── streamlit_app.py               # Web UI application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── data/
│   └── mtsamples.csv             # Medical transcriptions dataset
└── output/
    ├── faiss_medical_index/      # Vector store
    ├── preprocessed_chunks.pkl   # Document chunks
    ├── rag_config.pkl            # Configuration
    ├── evaluation_results.csv    # Evaluation metrics
    └── evaluation_report.md      # Detailed report
```

## 🎯 Usage

### Web Interface

1. Launch the Streamlit app
2. Type your medical question in the text area
3. Adjust settings in the sidebar (optional):
   - Number of sources to retrieve (1-10)
   - LLM temperature (0.0-1.0)
4. Click "Get Answer"
5. View the answer and source documents

### Example Questions

- "What are the symptoms of heart failure?"
- "How is a colonoscopy procedure performed?"
- "What medications are used to treat hypertension?"
- "What is the recovery time for hip replacement?"
- "What are the warning signs of a stroke?"

## 📊 Performance

**Evaluation Results (33 Medical Queries):**
- Success Rate: ~100%
- Average Answer Length: 200-400 characters
- Average Sources per Query: 4 documents
- Processing Time: ~2-5 seconds per query

**Medical Specialties Covered:**
- Cardiology
- Gastroenterology
- Orthopedics
- Neurology
- Pulmonology
- Dermatology
- Endocrinology
- General Surgery
- Urology
- Ophthalmology

## 🔧 Configuration

### Model Settings

Edit in `streamlit_app.py` or notebook:

```python
LLM_MODEL = "llama3.2"              # Ollama model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # HuggingFace model
CHUNK_SIZE = 1000                    # Document chunk size
CHUNK_OVERLAP = 200                  # Overlap between chunks
RETRIEVER_K = 4                      # Number of documents to retrieve
TEMPERATURE = 0.3                    # LLM temperature (0.0-1.0)
```

### Advanced Configuration

- **Change chunk size**: Larger chunks provide more context but may reduce precision
- **Adjust retrieval count**: More sources improve accuracy but increase processing time
- **Modify temperature**: Lower values (0.1-0.3) for factual responses, higher (0.6-0.8) for creative

## 🐳 Deployment

### Local Deployment

Already covered in Quick Start above.

### Cloud Deployment Options

#### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. **Note**: Requires Ollama accessible via network or API endpoint

#### Option 2: Docker Container

```dockerfile
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Ollama (for Linux)
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

Build and run:
```bash
docker build -t medical-rag .
docker run -p 8501:8501 medical-rag
```

#### Option 3: AWS/Azure/GCP

1. Set up a VM instance
2. Install Ollama and dependencies
3. Run Streamlit app as a service
4. Configure firewall for port 8501

## 🛠️ Development

### Rebuilding the Vector Store

If you modify the dataset or chunk settings:

```python
# Run these cells in the notebook:
# Step 5: Data Preprocessing
# Step 6: Create Embeddings and Vector Store
```

### Switching Models

**Different LLM:**
```bash
ollama pull mistral  # or llama3.1, phi3, etc.
```

Update `LLM_MODEL = "mistral"` in code.

**Different Embeddings:**
```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

## 📝 Requirements

See `requirements.txt` for full list:

```txt
langchain>=1.1.0
langchain-ollama>=1.0.0
langchain-community>=0.4.1
sentence-transformers>=5.1.2
faiss-cpu>=1.13.0
pandas>=2.3.3
numpy>=2.3.5
streamlit>=1.40.0
```

## ⚠️ Important Notes

1. **First Run**: Downloading embedding model takes ~80MB (one-time)
2. **Memory**: Requires ~4GB RAM for full dataset processing
3. **Ollama**: Must be running before starting the application
4. **Dataset**: Not included; download separately from Kaggle

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Add more evaluation metrics
- Implement conversation memory
- Support multiple languages
- Add medical fact-checking layer
- Optimize for GPU acceleration

## 📄 License

This project uses the Medical Transcriptions dataset from Kaggle, subject to its license terms.

## 🙏 Acknowledgments

- Medical Transcriptions dataset by [tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- LangChain framework
- Ollama team for local LLM hosting
- HuggingFace for embedding models

## 📧 Support

For issues or questions:
1. Check existing documentation
2. Review error messages in terminal
3. Verify Ollama is running: `ollama list`
4. Ensure dataset is in correct location

## 🔄 Version History

- **v1.0** (2025-11-27): Initial release
  - Complete RAG pipeline
  - Streamlit web interface
  - Comprehensive evaluation
  - Documentation

---

**Built with ❤️ for medical information retrieval**
