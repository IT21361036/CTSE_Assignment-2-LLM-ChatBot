# CTSE Lecture Notes Chatbot  📚

## Objective
Develop a simple chatbot using an LLM to answer questions based on CTSE lecture notes. The chatbot is implemented in a Jupyter Notebook. The project will also include a justification report and a short video demonstration.

A Streamlit-based chatbot that answers questions using CTSE lecture notes via OpenRouter's LLM API.
# Features ✨
- Upload and query PDF/PPTX lecture notes
- Semantic search using TF-IDF vectors
- Conversation history with Mistral-7B-instruct model
- Secure API key management

## Tech Stack 🛠️
- **Backend**: Python
- **LLM**: `mistralai/mistral-7b-instruct` via OpenRouter
- **Libraries**: 
  - Streamlit (UI)
  - LangChain (LLM integration)
  - scikit-learn (semantic search)
  - PyPDF2/pptx (document processing)

## Setup Instructions 🚀

### 1. Prerequisites
- Python 3.9+
- Git (for version control)

### 2. Installation
```
bash
# Clone repository
git clone https://github.com/yourusername/ctse-llm-chatbot.git
cd ctse-llm-chatbot
```

# Create virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```
# Install dependencies
```
pip install -r requirements.txt
```
