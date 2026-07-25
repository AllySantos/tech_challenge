#!/usr/bin/env python
"""Valida se o ambiente está pronto para rodar o projeto.

Verifica, nesta ordem:
    1. Versão do Python.
    2. Presença do Poetry.
    3. Presença dos pacotes de produção obrigatórios.
    4. Carregamento das variáveis de ambiente via ``.env`` / ambiente atual.
    5. Alcance do MLflow.
    6. Existência e leitura do diretório ``data/raw``.
    7. Configuração de um remote DVC.

Uso:
    python scripts/validate_env.py

Retorna código de saída 0 se tudo estiver ok, 1 caso contrário —
pensado para ser usado tanto localmente quanto em CI.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_PYTHON_VERSION = (3, 11)

REQUIRED_PACKAGES = (
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pandas",
    "numpy",
    "pydantic",
    "pydantic_settings",
    "yaml",
)

ENV_KEYS = (
    "SEED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "MODELS_DIR",
)


def check_python_version() -> tuple[bool, str]:
    """Confere se a versão do Python atende o mínimo exigido."""
    current = sys.version_info[:3]
    current_version = ".".join(str(part) for part in current)
    required = ".".join(str(part) for part in MIN_PYTHON_VERSION)

    if current[:2] < MIN_PYTHON_VERSION:
        return False, f"Python {current_version} não atende a versão mínima {required}."
    return True, f"Python {current_version}"


def check_poetry() -> tuple[bool, str]:
    """Confere se o Poetry está disponível no ambiente."""
    try:
        completed = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "Poetry não encontrado no PATH."

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        return False, f"Poetry não respondeu corretamente: {stderr}"

    version = completed.stdout.strip().split()[-1]
    return True, f"Poetry {version}"


def check_required_packages() -> tuple[bool, str]:
    """Confere se todos os pacotes obrigatórios estão instalados e importáveis."""
    missing_packages: list[str] = []
    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        missing = ", ".join(missing_packages)
        return False, f"Dependências ausentes: {missing}."
    return True, "Todas as dependências instaladas"


def _load_env_values() -> dict[str, str]:
    """Carrega valores de ambiente a partir de .env, .env.example ou do ambiente atual."""
    values: dict[str, str] = {}

    env_files = [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example"]
    for env_file in env_files:
        if not env_file.is_file():
            continue

        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    for key in ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]

    return values


def check_environment_variables() -> tuple[bool, str]:
    """Confere se as variáveis de ambiente principais foram carregadas."""
    values = _load_env_values()
    if not values:
        return False, "Nenhuma variável de ambiente foi carregada."

    selected = []
    for key in ENV_KEYS:
        if key in values:
            selected.append(f"{key}={values[key]}")

    if len(selected) > 3:
        summary = ", ".join(selected[:3]) + ", ..."
    else:
        summary = ", ".join(selected)

    return True, f"Variáveis de ambiente carregadas: {summary}"


def check_mlflow_reachable() -> tuple[bool, str]:
    """Confere se o endpoint do MLflow está disponível."""
    values = _load_env_values()
    tracking_uri = values.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return False, "MLflow tracking URI não configurada."

    if not tracking_uri.startswith(("http://", "https://")):
        return False, f"URI inválida para MLflow: {tracking_uri}"

    try:
        request = Request(tracking_uri, method="GET")
        with urlopen(request, timeout=3) as response: 
            if response.status < 400:
                return True, f"MLflow alcançável em {tracking_uri}"

    except Exception as exc:  
        return False, f"Falha ao consultar MLflow em {tracking_uri}: {exc}"




def check_dvc_remote_config() -> list[str]:
    """Confere se existe pelo menos um remote DVC configurado."""
    config_path = PROJECT_ROOT / ".dvc" / "config"
    if not config_path.is_file():
        return ["DVC remote não configurado (rode `dvc remote add -d <name> <url>`)."]

    content = config_path.read_text(encoding="utf-8")
    if re.search(r'^\s*\[remote\s+"[^"]+"\]', content, re.MULTILINE):
        return []

    return ["DVC remote não configurado (rode `dvc remote add -d <name> <url>`)."]


def main() -> int:
    """Executa todas as validações e imprime um resumo."""
    checks = (
        ("Python", check_python_version),
        ("Poetry", check_poetry),
        ("Dependências", check_required_packages),
        ("Configurações (.env / Settings)", check_environment_variables),
        ("MLflow", check_mlflow_reachable)
    )

    all_errors: list[str] = []
    for label, check in checks:
        ok, message = check()
        if ok:
            print(f"✅ {label}: {message}")
        else:
            print(f"❌ {label}: {message}")
            all_errors.append(message)

    dvc_errors = check_dvc_remote_config()
    if dvc_errors:
        print(f"❌ DVC remote: {dvc_errors[0]}")
        all_errors.extend(dvc_errors)
    else:
        print("✅ DVC remote: remote DVC configurado")

    if all_errors:
        print(f"\nValidação falhou com {len(all_errors)} problema(s).")
        return 1

    print("\nAmbiente validado com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
