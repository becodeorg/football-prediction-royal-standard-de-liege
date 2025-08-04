from src.backend.data_pipeline import run_pipeline
from src.backend.trainer_pipeline import run_training_pipeline
from utils.dropbox_io import upload_model_to_dropbox
from config import settings


def main(source_name: str = "B1_old") -> None:
    run_pipeline(source_name=source_name, save=True)
    run_training_pipeline(source_name=source_name)


if __name__ == "__main__":
    main()
