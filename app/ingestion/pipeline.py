class IngestionPipeline:
    def __init__(
            self,
            discovery,
            loader,
            chunker,
            ingestion_service
    ) -> None:
        self.discovery = discovery
        self.loader = loader
        self.chunker = chunker
        self.ingestion_service = ingestion_service

    def run(self) -> None:
        files = self.discovery.find()

        for path in files:
            document = self.loader.load(path)
            chunks = self.chunker.split(document)

            for chunk in chunks:
                self.ingestion_service.ingest(chunk=chunk)
