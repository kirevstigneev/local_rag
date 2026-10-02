from app.ingestion.discovery import DocumentDiscovery


def test_discovery_returns_txt_files(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    first = documents_dir / "first.txt"
    second = documents_dir / "second.txt"
    ignored = documents_dir / "ignored.pdf"

    first.write_text("First document.", encoding="utf-8")
    second.write_text("Second document.", encoding="utf-8")
    ignored.write_text("PDF document.", encoding="utf-8")

    discovery = DocumentDiscovery(documents_dir)

    files = discovery.find()

    assert files == [first, second]
