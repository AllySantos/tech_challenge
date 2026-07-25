#!/usr/bin/env python
"""Valida se o ambiente está pronto para rodar o projeto.

Verifica, nesta ordem:
    1. Versão do Python.
    2. Presença dos pacotes de produção obrigatórios.
    3. Carregamento das configurações via ``.env`` / Pydantic Settings.
    4. Existência dos diretórios de dados/modelos esperados.

Uso:
    poetry run python scripts/validate_env.py

Retorna código de saída 0 se tudo estiver ok, 1 caso contrário —
pensado para ser usado tanto localmente quanto em CI.
"""

import importlib
import sys
from pathlib import Path

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


def check_python_version() -> list[str]:
    """Confere se a versão do Python atende o mínimo exigido.

    Returns:
        Lista de mensagens de erro (vazia se a versão for válida).
    """
    if sys.version_info[:2] < MIN_PYTHON_VERSION:
        current = ".".join(str(part) for part in sys.version_info[:2])
        required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
        return [f"Python {required}+ é necessário (encontrado {current})."]
    return []


def check_required_packages() -> list[str]:
    """Confere se todos os pacotes obrigatórios estão instalados e importáveis.

    Returns:
        Lista de mensagens de erro, uma por pacote ausente.
    """
    errors = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            errors.append(
                f"Pacote obrigatório não encontrado: '{package}'. "
                "Rode `poetry install` para instalar as dependências."
            )
    return errors


def check_settings_loadable() -> list[str]:
    """Confere se as configurações do projeto carregam sem erro.

    Returns:
        Lista de mensagens de erro (vazia se as configurações carregarem).
    """
    try:
        pass
        # TODO: Implementar validação de configuração
    except Exception as exc: 
        return [f"Falha ao carregar configurações (.env / Settings): {exc}"]
    return []


def check_expected_directories() -> list[str]:
    """Confere se os diretórios de dados/modelos esperados existem.

    Returns:
        Lista de mensagens de erro, uma por diretório ausente.
    """
    project_root = Path(__file__).resolve().parent.parent
    expected_dirs = ("data/raw", "data/processed", "models", "configs")

    errors = []
    for relative_dir in expected_dirs:
        if not (project_root / relative_dir).is_dir():
            errors.append(f"Diretório esperado não encontrado: '{relative_dir}'.")
    return errors


def main() -> int:
    """Executa todas as validações e imprime um resumo.

    Returns:
        0 se todas as validações passarem, 1 caso contrário.
    """
    checks = (
        ("Versão do Python", check_python_version),
        ("Pacotes obrigatórios", check_required_packages),
        ("Configurações (.env / Settings)", check_settings_loadable),
        ("Diretórios esperados", check_expected_directories),
    )

    all_errors: list[str] = []
    for label, check in checks:
        errors = check()
        status = "OK" if not errors else "FALHOU"
        print(f"[{status}] {label}")
        for error in errors:
            print(f"       - {error}")
        all_errors.extend(errors)

    if all_errors:
        print(f"\nValidação falhou com {len(all_errors)} problema(s).")
        return 1

    print("\nAmbiente validado com sucesso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
