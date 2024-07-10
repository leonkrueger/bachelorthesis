import os


def load_env_variables() -> None:
    with open(os.path.join("..", ".env")) as env_file:
        for line in env_file.readlines():
            if line.startswith("#") or not line.strip():
                continue

            key, value = line.strip().split("=", 1)
            os.environ[key] = value
