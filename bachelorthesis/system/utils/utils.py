import logging
import os
from pathlib import Path


def load_env_variables() -> None:
    """Loads the environment variables stored in the .env file.
    File needs to be in the top folder of the project."""
    with open(Path(__file__).resolve().parents[3] / ".env") as env_file:
        for line in env_file.readlines():
            if line.startswith("#") or not line.strip():
                continue

            key, value = line.strip().split("=", 1)
            os.environ[key] = value


def configure_logger(logging_file: str) -> None:
    """Configures logger"""
    logging.basicConfig(
        filename=logging_file,
        level=logging.INFO,
        encoding="utf-8",
    )


def get_finetuned_model_dir(name: str) -> os.PathLike:
    """Returns the path of the finetuned model in this project"""
    return (
        Path(__file__).resolve().parents[3] / os.environ["FINETUNED_MODELS_DIR"] / name
    )


def remove_quotes(name: str) -> str:
    """Removes the quotes from a name, if it has any"""
    if name.startswith("'") or name.startswith('"') or name.startswith("`"):
        return name[1:-1]
    return name
