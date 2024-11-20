"""
Contextual RAG system implementation with Llama model and chunk-based processing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextualRAG:
    MAX_CONTEXT_SIZE = 8192  # Default max context size for Llama models
    CHUNK_SIZE_FACTOR = 0.9  # 10% less than max context size

    def __init__(self, data_dir: str, persist_dir: str = "./chroma_db"):
        """
        Initialize ContextualRAG system.

        Args:
            data_dir: Directory containing the source files
            persist_dir: Directory to persist Chroma DB
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.model = ChatOllama(model="llama2:7b-4bit")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.chunk_metadata: Dict[str, Dict] = {}
        self.vectorstore = None

    def get_context_size(self) -> int:
        """Get the maximum context size for the model."""
        # In a real implementation, you might query the model for this
        return self.MAX_CONTEXT_SIZE

    def load_and_split_files(self) -> List[Tuple[str, str, int]]:
        """
        Load files from data directory and split into chunks.

        Returns:
            List of tuples containing (chunk_text, filename, position)
        """
        chunks_with_metadata = []
        chunk_size = int(self.get_context_size() * self.CHUNK_SIZE_FACTOR)

        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=200, length_function=len
        )

        for file_path in self.data_dir.glob("*.*"):
            if file_path.is_file():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    chunks = text_splitter.split_text(content)

                    for position, chunk in enumerate(chunks):
                        chunks_with_metadata.append((chunk, file_path.name, position))

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

        return chunks_with_metadata

    def create_chunk_prefix(self, chunk: str, filename: str, position: int) -> str:
        """
        Create a prefix for each chunk using the model.

        Args:
            chunk: The text chunk
            filename: Source filename
            position: Position in the file

        Returns:
            Generated prefix
        """
        prompt_template = """Create a prefix for this chunk:
        
Content: {chunk}

The prefix should include:
- File: {filename}
- Position: {position}
- Summary of content (AI-readable)

Generate a concise prefix:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model | StrOutputParser()

        try:
            prefix = chain.invoke(
                {"chunk": chunk, "filename": filename, "position": position}
            )
            return prefix.strip()
        except Exception as e:
            logger.error(f"Error creating prefix: {str(e)}")
            return f"File: {filename}, Position: {position}"

    def process_and_embed(self):
        """Process all files, create prefixes, and embed in Chroma DB."""
        chunks_with_metadata = self.load_and_split_files()
        processed_chunks = []

        for chunk, filename, position in chunks_with_metadata:
            prefix = self.create_chunk_prefix(chunk, filename, position)
            processed_text = f"{prefix}\n\n{chunk}"
            processed_chunks.append(processed_text)

            # Store metadata
            chunk_id = f"{filename}_{position}"
            self.chunk_metadata[chunk_id] = {
                "filename": filename,
                "position": position,
                "prefix": prefix,
            }

        # Create and persist vector store
        self.vectorstore = Chroma.from_texts(
            texts=processed_chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self.vectorstore.persist()

    def infer(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Infer using RAG with the user prompt.

        Args:
            query: User query
            top_k: Number of most similar chunks to return

        Returns:
            List of relevant chunks with their metadata
        """
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Run process_and_embed first."
            )

        # Get similar chunks using cosine similarity
        results = self.vectorstore.similarity_search_with_scores(query, k=top_k)

        response = []
        for doc, score in results:
            # Extract filename and position from the content
            content_lines = doc.page_content.split("\n")
            metadata = {}

            # Find matching chunk metadata
            for chunk_id, meta in self.chunk_metadata.items():
                if meta["prefix"] in doc.page_content:
                    metadata = meta
                    break

            response.append(
                {
                    "content": doc.page_content,
                    "similarity_score": score,
                    "metadata": metadata,
                }
            )

        return response

    def rag_create(self):
        """
        Initialize or update the RAG system by processing files and updating Chroma DB.
        """
        logger.info("Starting RAG creation process...")
        logger.info(f"Reading files from: {self.data_dir}")

        try:
            self.process_and_embed()
            logger.info("Successfully created and persisted RAG system")
            logger.info(f"Processed chunks: {len(self.chunk_metadata)}")
            logger.info(f"Chroma DB location: {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error during RAG creation: {str(e)}")
            raise


def main():
    """
    Main function to run the RAG system with interactive user prompts.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Contextual RAG System")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing source files",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_db",
        help="Directory to persist Chroma DB",
    )
    parser.add_argument(
        "--create", action="store_true", help="Create/update the RAG system"
    )
    args = parser.parse_args()

    # Initialize RAG system
    rag = ContextualRAG(data_dir=args.data_dir, persist_dir=args.persist_dir)

    # Create/update RAG if requested
    if args.create:
        logger.info("Creating/updating RAG system...")
        rag.rag_create()
        logger.info("RAG system ready!")
        return

    # Check if vectorstore exists
    if not os.path.exists(args.persist_dir):
        logger.error("No existing RAG system found. Please run with --create first.")
        return

    # Interactive query loop
    print("\nContextual RAG System")
    print("Enter your queries (type 'exit' to quit)")
    print("-" * 50)

    while True:
        try:
            query = input("\nEnter your query: ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                break

            if not query:
                continue

            results = rag.infer(query)

            print("\nResults:")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"File: {result['metadata'].get('filename', 'N/A')}")
                print(f"Position: {result['metadata'].get('position', 'N/A')}")
                print(f"Similarity Score: {result['similarity_score']:.4f}")
                print("\nContent Preview:")
                print("-" * 30)
                # Print first 200 characters of content
                print(f"{result['content'][:200]}...")
                print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print("An error occurred. Please try again.")


if __name__ == "__main__":
    main()
