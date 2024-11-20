from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class RAGAssistant:
    def __init__(
        self, model_name: str = "mistral", embedding_model: str = "nomic-embed-text"
    ):
        """Initialize RAG Assistant with specified models."""
        self.model_local = ChatOllama(model=model_name)
        self.embedding_model = embeddings.ollama.OllamaEmbeddings(model=embedding_model)
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=7500, chunk_overlap=100
        )
        self.vectorstore = None
        self.retriever = None

    def load_from_urls(self, urls: List[str]) -> None:
        """Load and process documents from URLs."""
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        doc_splits = self.text_splitter.split_documents(docs_list)
        self._create_vectorstore(doc_splits)

    def load_from_pdf(self, pdf_path: str) -> None:
        """Load and process documents from a PDF file."""
        loader = PyPDFLoader(pdf_path)
        doc_splits = loader.load_and_split()
        self._create_vectorstore(doc_splits)

    def _create_vectorstore(self, documents) -> None:
        """Create vector store from documents."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            collection_name="rag-chroma",
            embedding=self.embedding_model,
        )
        self.retriever = self.vectorstore.as_retriever()

    def query_without_context(self, topic: str) -> str:
        """Query the model without RAG context."""
        template = "What is {topic}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model_local | StrOutputParser()
        return chain.invoke({"topic": topic})

    def query_with_context(self, question: str) -> str:
        """Query the model with RAG context."""
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")

        template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.model_local
            | StrOutputParser()
        )
        return chain.invoke(question)


def main():
    # Example usage
    urls = [
        "https://ollama.com/",
        "https://ollama.com/blog/windows-preview",
        "https://ollama.com/blog/openai-compatibility",
        "https://antongorlin.com/blog/photography-composition-definitive-guide",
    ]

    # Initialize RAG Assistant
    rag = RAGAssistant()

    # Load documents from URLs
    rag.load_from_urls(urls)

    # Example queries
    print("Before RAG\n")
    print(rag.query_without_context("Ollama"))

    print("\n########\nAfter RAG\n")
    print(rag.query_with_context("What is Ollama?"))


if __name__ == "__main__":
    main()
