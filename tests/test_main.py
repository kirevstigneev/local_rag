from unittest.mock import Mock, patch

from app.ingestion.main import main



def test_main_runs_ingestion_pipeline():
    pipeline = Mock()

    with patch(
        "app.ingestion.main.create_ingestion_pipeline",
        return_value=pipeline,
    ):
        main()

    pipeline.run.assert_called_once_with()
