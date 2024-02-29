import os
import logging
from elasticsearch import Elasticsearch, ElasticsearchException
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedRAG:
    def __init__(self, es_host='localhost', es_port=9200, model_name='gpt2', use_semantic_search=False, documents_dir=None):
        self.use_semantic_search = use_semantic_search
        if use_semantic_search:
            if documents_dir:
                self.documents = self.load_documents(documents_dir)
                try:
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logging.error(f"Failed to load Sentence Transformer model: {e}")
                    raise
            else:
                logging.error("Documents directory must be provided for semantic search.")
                raise ValueError("Documents directory not provided.")
        else:
            try:
                self.es = Elasticsearch([{'host': es_host, 'port': es_port}])
                if not self.es.ping():
                    logging.error("Connection to Elasticsearch failed!")
                    raise ConnectionError("Failed to connect to Elasticsearch.")
            except ElasticsearchException as e:
                logging.error(f"Elasticsearch connection error: {e}")
                raise

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        except Exception as e:
            logging.error(f"Failed to load GPT-2 model: {e}")
            raise

    def load_documents(self, directory):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        documents.append(file.read())
                except IOError as e:
                    logging.warning(f"Failed to read file {filepath}: {e}")
        return documents

    def retrieve_documents_es(self, query, index="documents", top_n=1):
        try:
            response = self.es.search(index=index, body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": top_n
            })
            documents = [hit["_source"]["content"] for hit in response["hits"]["hits"]]
            return documents
        except ElasticsearchException as e:
            logging.error(f"Error retrieving documents from Elasticsearch: {e}")
            return []

    def retrieve_documents_semantic(self, query):
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.semantic_model.encode(self.documents, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        most_relevant_document_index = cosine_scores.argmax()
        return self.documents[most_relevant_document_index]

    def generate_response(self, input_text, max_length=100, temperature=0.8):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output_sequences = self.model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

def interactive_mode(rag):
    print("Enter your query or type 'exit' to quit.")
    while True:
        query = input("> ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break

        if rag.use_semantic_search:
            retrieved_document = rag.retrieve_documents_semantic(query)
        else:
            retrieved_documents = rag.retrieve_documents_es(query)
            retrieved_document = retrieved_documents[0] if retrieved_documents else "No relevant document found."

        input_text_for_generation = retrieved_document + " " + query
        generated_response = rag.generate_response(input_text_for_generation)

        print(f"Generated Response: {generated_response}\n")

        # Collect and handle user feedback for future improvements
        feedback = input("Was this response helpful? (yes/no): ")
        logging.info(f"Query: {query}, Helpful: {feedback}")

if __name__ == "__main__":
    directory = os.getenv('DOCUMENTS_DIR', 'path/to/your/documents')
    es_host = os.getenv('ES_HOST', 'localhost')
    es_port = int(os.getenv('ES_PORT', 9200))
    use_semantic_search = os.getenv('USE_SEMANTIC_SEARCH', 'False').lower() in ['true', '1', 't']

    try:
        rag = AdvancedRAG(es_host=es_host, es_port=es_port, use_semantic_search=use_semantic_search, documents_dir=directory)
        interactive_mode(rag)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
