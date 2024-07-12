import os


def load_env_variables() -> None:
    """Loads the environment variables stored in the .env file.
    File needs to be in the top folder of the project."""
    with open(os.path.join("..", "..", "..", ".env")) as env_file:
        for line in env_file.readlines():
            if line.startswith("#") or not line.strip():
                continue

            key, value = line.strip().split("=", 1)
            os.environ[key] = value


def remove_quotes(name: str) -> str:
    """Removes the quotes from a name, if it has any"""
    if name.startswith("'") or name.startswith('"'):
        return name[1:-1]
    return name
