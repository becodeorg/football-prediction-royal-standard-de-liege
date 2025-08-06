from src.backend.data_pipeline import run_pipeline
from src.backend.trainer_pipeline import run_training_pipeline
from utils.dropbox.dropbox_io import upload_model_to_dropbox
from config import settings


def main(source_name: str = "Belgium_league_2526") -> None:
    run_pipeline(source_name=source_name, save=True)

    model = run_training_pipeline(source_name=source_name, save=True)

    upload_model_to_dropbox(
        model=model,
        dropbox_path=settings.dropbox_model_path
    )


if __name__ == "__main__":
    main()
