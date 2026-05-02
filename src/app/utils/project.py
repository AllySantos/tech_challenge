import tomllib


class Project:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version


def get_project_info() -> Project:
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    project_info = data.get("project", {})

    name = project_info.get("name", "churn-prediction")
    version = project_info.get("version", "0.1.0")

    return Project(name=name, version=version)
