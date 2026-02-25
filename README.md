# RAG Chatbot with Groq and Ollama

A powerful RAG (Retrieval-Augmented Generation) Chatbot built with **Streamlit**, **LangChain (LCEL)**, **Groq (LLama 3.1/3.2)**, and **Ollama Embeddings**.

## 🚀 Features
- **LCEL Architecture**: Uses the modern LangChain Expression Language for robust and modular RAG flows.
- **High Performance**: Leverages Groq's high-speed inference for near-instant responses.
- **Multi-PDF Support**: Automatically loads and indexes all PDF research papers in the `research_paper/` directory.
- **Vector Search**: Uses FAISS for efficient similarity search across document chunks.

## 🛠️ Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Pull the Embedding Model**:
   Make sure you have your preferred embedding model (e.g., `gemma:2b` or `llama3`) pulled in Ollama:
   ```bash
   ollama pull gemma:2b
   ```

## 🏃 Usage
1. Place your PDF documents in the `research_paper/` folder.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Click the **"Vector Embeddings"** button to index your documents.
4. Start asking questions about your research papers!

## 📂 Project Structure
- `app.py`: Main Streamlit application with LCEL RAG logic.
- `research_paper/`: Folder containing the PDF source documents.
- `requirements.txt`: Python package dependencies.
- `.env`: Environment variables (API keys).
- `.gitignore`: Files excluded from version control.
