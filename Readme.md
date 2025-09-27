# Multimodal RAG with MongoDB

<img width="800" height="378" alt="image" src="https://github.com/user-attachments/assets/452b5a07-bc26-4206-a8da-bfb63cc020f5" />

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system using MongoDB as a vector store. It enables users to upload PDF documents, extract both text and image content, index them with embeddings, and interactively ask questions about the content via a Streamlit web interface.

## Features

- **PDF Upload:** Upload PDF files for processing.
- **Text and Image Extraction:** Extracts text and images (including tables) from PDFs. Images are processed using OCR and LLM-based vision models.
- **Chunking:** Splits extracted content into manageable chunks for efficient retrieval.
- **Vector Store:** Stores document embeddings in MongoDB using Atlas Vector Search.
- **Question Answering:** Users can ask questions about the uploaded documents; the system retrieves relevant chunks and generates concise answers using an LLM.
- **Streamlit UI:** Simple web interface for interaction.

## Installation

### Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available on your system (update the path in the code if needed)
- MongoDB running locally or accessible remotely

### Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
streamlit
langchain
langchain-core
langchain-community
langchain-mongodb
langchain-ollama
langchain-text-splitters
pymongo
unstructured[pdf]
```

You may need to install additional dependencies for PDF/image processing as prompted.

### Configuration

- Ensure the `pdfs/` and `pdf_figures/` directories exist in the project root.
- Update the Tesseract OCR path in the code if your installation is elsewhere.
- Ensure MongoDB is running and accessible at the URI specified in the code.

### Running the App

```bash
streamlit run multimodal_mongo.py
```


## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF file.
3. Wait for processing and indexing.
4. Enter your question in the chat input.
5. View the assistant's concise answer based on the document content.

## Notes

- The system uses HuggingFace or Ollama embeddings and LLMs. Ensure you have access to the required models.
- For large PDFs or many images, processing may take some time.
- MongoDB Atlas Vector Search requires proper setup if using a remote database.

## License

MIT License

