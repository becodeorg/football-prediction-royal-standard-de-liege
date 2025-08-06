import io
import logging
from typing import Any

import joblib
import dropbox
import dropbox.exceptions
from dropbox.files import WriteMode

from config import settings
from utils.dropbox.token_manager import refresh_access_token

logger = logging.getLogger(__name__)


def get_dropbox_client():
    """
    Return Dropbox client. If token expired, refresh and retry.
    """
    try:
        dbx = dropbox.Dropbox(settings.dropbox_access_token)
        dbx.users_get_current_account()
        return dbx
    except dropbox.exceptions.AuthError:
        logger.warning("Dropbox token expired or invalid, refreshing...")
        refresh_access_token()
        dbx = dropbox.Dropbox(settings.dropbox_access_token)
        dbx.users_get_current_account()
        return dbx


def upload_model_to_dropbox(model: Any, dropbox_path: str) -> None:
    """
    Upload a serialized model object to Dropbox at the specified path.

    :param model: Trained model object.
    :param dropbox_path: Path in Dropbox.
    :raises: dropbox.exceptions.ApiError: If there is an API error during upload.
             Exception: For any other unexpected errors.
    """
    dbx = get_dropbox_client()
    try:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        dbx.files_upload(
            buffer.read(),
            dropbox_path,
            mode=WriteMode("overwrite")
        )
        logger.info(f"Model uploaded to Dropbox at {dropbox_path}")
    except dropbox.exceptions.ApiError as api_err:
        logger.error(f"Dropbox API error during upload: {api_err}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise


def download_model_from_dropbox(dropbox_path: str) -> Any:
    """
    Download and deserialize a model object from Dropbox.

    :param dropbox_path: Path to model in Dropbox.
    :return: Loaded model object.
    :raises: dropbox.exceptions.HttpError: If there is an HTTP error during download.
             Exception: For any other unexpected errors.
    """
    dbx = get_dropbox_client()
    try:
        metadata, res = dbx.files_download(dropbox_path)
        buffer = io.BytesIO(res.content)
        model = joblib.load(buffer)
        logger.info(f"Model downloaded from Dropbox: {dropbox_path}")
        return model
    except dropbox.exceptions.HttpError as http_err:
        logger.error(f"Dropbox HTTP error during download: {http_err}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise
