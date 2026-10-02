from app.container import create_ingestion_pipeline


def main() -> None:
    pipeline = create_ingestion_pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
