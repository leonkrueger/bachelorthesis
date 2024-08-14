import logging
import os


def load_env_variables() -> None:
    """Loads the environment variables stored in the .env file.
    File needs to be in the top folder of the project."""
    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", ".env"
        )
    ) as env_file:
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


def remove_quotes(name: str) -> str:
    """Removes the quotes from a name, if it has any"""
    if name.startswith("'") or name.startswith('"'):
        return name[1:-1]
    return name
