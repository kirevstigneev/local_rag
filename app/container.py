from app.rag.context import ContextBuilder
from app.ingestion.chunking import TextChunker
from app.core.config import settings
from app.ingestion.discovery import DocumentDiscovery
from app.clients.embeddings import EmbeddingClient
from app.retrieval.hybrid_retrieval import HybridRetriever
from app.ingestion.ingestion import IngestionService
from app.clients.llm import LLMClient
from app.ingestion.loader import TextFileLoader
from app.ingestion.pipeline import IngestionPipeline
from app.clients.qdrant import QdrantRepository
from app.rag.rag import RAGService
from app.retrieval.reranking import Reranker
from app.retrieval.retrieval import Retriever
from app.retrieval.lexical_retrieval import LexicalRetriever


def create_ingestion_pipeline() -> IngestionPipeline:
    discovery = DocumentDiscovery(
        documents_dir=settings.documents_dir,
    )

    loader = TextFileLoader()
    chunker = TextChunker()

    embedding_client = EmbeddingClient()
    qdrant_repository = QdrantRepository()

    qdrant_repository.ensure_collection()

    ingestion_service = IngestionService(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    return IngestionPipeline(
        discovery=discovery,
        loader=loader,
        chunker=chunker,
        ingestion_service=ingestion_service
    )


def create_rag_service() -> RAGService:
    embedding_client = EmbeddingClient()
    qdrant_repository = QdrantRepository()

    vector_retriever = Retriever(
        embedding_client=embedding_client,
        qdrant_repository=qdrant_repository,
    )

    chunks = qdrant_repository.get_all_chunks()
    lexical_retriever = LexicalRetriever(
        chunks=chunks,
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lexical_retriever=lexical_retriever,
    )

    reranker = Reranker(
        model_name=settings.reranker_model,
    )

    llm_client = LLMClient()
    context_builder = ContextBuilder()

    return RAGService(
        retriever=hybrid_retriever,
        llm_client=llm_client,
        context_builder=context_builder,
        reranker=reranker
    )
