from unittest.mock import Mock

from app.core.models import Document
from app.ingestion.pipeline import IngestionPipeline


def test_pipeline_loads_documents_from_discovered_files(tmp_path):
    discovery = Mock()
    loader = Mock()
    chunker = Mock()
    ingestion_service = Mock()

    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("First document.", encoding="utf-8")
    second_file.write_text("Second document.", encoding="utf-8")

    discovery.find.return_value = [
        first_file,
        second_file
    ]

    first_document = Document(
        id=1,
        text="First document.",
        source=str(first_file),
        metadata={},
    )

    second_document = Document(
        id=2,
        text="Second document.",
        source=str(second_file),
        metadata={}
    )

    loader.load.side_effect = [
        first_document,
        second_document
    ]

    chunker.split.return_value = []

    pipeline = IngestionPipeline(
        discovery=discovery,
        loader=loader,
        chunker=chunker,
        ingestion_service=ingestion_service
    )

    pipeline.run()

    discovery.find.assert_called_once_with()

    assert loader.load.call_count == 2
    loader.load.assert_any_call(first_file)
    loader.load.assert_any_call(second_file)


def test_pipeline_chunks_loaded_documents(tmp_path):
    discovery = Mock()
    loader = Mock()
    chunker = Mock()
    ingestion_service = Mock()

    file = tmp_path / "document.txt"
    file.write_text(
        "Qdrant is a vector database.",
        encoding="utf-8",
    )

    document = Document(
        id=1,
        text="Qdrant is a vector database.",
        source=str(file),
        metadata={}
    )

    chunks = [
        Mock(),
        Mock(),
    ]

    discovery.find.return_value = [file]
    loader.load.return_value = document
    chunker.split.return_value = chunks

    pipeline = IngestionPipeline(
        discovery=discovery,
        loader=loader,
        chunker=chunker,
        ingestion_service=ingestion_service
    )

    pipeline.run()

    chunker.split.assert_called_once_with(document)


def test_pipeline_ingests_all_chunks(tmp_path):
    discovery = Mock()
    loader = Mock()
    chunker = Mock()
    ingestion_service = Mock()

    file = tmp_path / "document.txt"
    file.write_text(
        "Qdrant is a vector database.",
        encoding="utf-8",
    )

    document = Document(
        id=1,
        text="Qdrant is a vector database.",
        source=str(file),
        metadata={},
    )

    first_chunk = Mock()
    second_chunk = Mock()

    discovery.find.return_value = [file]
    loader.load.return_value = document
    chunker.split.return_value = [
        first_chunk,
        second_chunk
    ]

    pipeline = IngestionPipeline(
        discovery=discovery,
        loader=loader,
        chunker=chunker,
        ingestion_service=ingestion_service
    )

    pipeline.run()

    ingestion_service.ingest.assert_any_call(
        chunk=first_chunk,
    )

    ingestion_service.ingest.assert_any_call(
        chunk=second_chunk,
    )

    assert ingestion_service.ingest.call_count == 2
