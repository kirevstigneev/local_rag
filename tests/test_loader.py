from pathlib import Path

from app.ingestion.loader import TextFileLoader


def test_loader_returns_document(tmp_path):
    file = tmp_path / "text.txt"

    file.write_text(
        "Qdrant is a vector database.",
        encoding="utf-8",
    )

    loader = TextFileLoader()

    document = loader.load(file)

    assert document.text == "Qdrant is a vector database."
    assert document.source == str(file)
    assert isinstance(document.id, int)
    assert document.metadata == {}


def test_document_id_is_determenistic(tmp_path):
    file = tmp_path / "text.txt"

    file.write_text(
            "Qdrant is a vector database.",
            encoding="utf-8",
        )

    loader = TextFileLoader()

    first = loader.load(file)
    second = loader.load(file)

    assert first.id == second.id


def test_defferent_files_have_different_ids(tmp_path):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("First document.", encoding="utf-8")
    second_file.write_text("Second docuemnt.", encoding="utf-8")

    loader = TextFileLoader()

    first = loader.load(first_file)
    second = loader.load(second_file)

    assert first.id != second.id
