# Advanced RAG System

This repository contains an implementation of an Advanced Retrieval-Augmented Generation (RAG) System, which leverages Elasticsearch for efficient document retrieval and a combination of Sentence Transformers and GPT-2 for generating contextually relevant responses based on a given query.

## Prerequisites

Before running the script, ensure you have:
- Python 3.x installed
- Elasticsearch running locally or accessible remotely
- Documents indexed in Elasticsearch (if not using semantic search)
- Internet connection (for downloading models and using Sentence Transformers)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Arkay92/Advanced-RAG-System.git
```

### Navigate to the cloned directory:
```bash
cd Advanced-RAG-System
```
### Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Configuration
The script uses environment variables for configuration. You can set these variables in your environment or directly in the script:

ES_HOST: Hostname of the Elasticsearch server (default: 'localhost')
ES_PORT: Port number of the Elasticsearch server (default: 9200)
USE_SEMANTIC_SEARCH: Whether to use semantic search instead of Elasticsearch ('true' or 'false', default: 'false')
DOCUMENTS_DIR: Directory containing documents for semantic search (required if USE_SEMANTIC_SEARCH is 'true')

## Usage
Run the script in interactive mode by executing:

```bash
python advanced_rag.py
```

Follow the prompts to enter queries and receive generated responses. Type 'exit' or 'quit' to terminate the interactive session.

## Contributing
Contributions to enhance or extend the functionality of this RAG system are welcome. Please feel free to submit pull requests or open issues with your suggestions or enhancements.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
