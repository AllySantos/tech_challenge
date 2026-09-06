"""Registry de modelos em disco, versionado por timestamp.

Cada retreino grava seus artefatos em ``models/<versao>/`` e só passa a ser
servido quando é promovido — momento em que ``models/current.json`` é
reescrito apontando para a nova versão. Isso mantém o rollback trivial (basta
promover a versão anterior) sem exigir um serviço de registry externo.

Um arquivo ponteiro é usado no lugar de um symlink porque o repositório é
compartilhado entre macOS, Linux e Windows.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.configs.settings import settings

logger = logging.getLogger(__name__)

POINTER_FILENAME = "current.json"
METADATA_FILENAME = "metadata.json"

PIPELINE_ARTIFACT = "pipeline.joblib"
PRUNED_PIPELINE_ARTIFACT = "pipeline.pruned.joblib"
ONNX_ARTIFACT = "model.onnx"
ONNX_INT8_ARTIFACT = "model.int8.onnx"
ONNX_PRUNED_ARTIFACT = "model.pruned.onnx"


def new_version(root: Path | None = None) -> Path:
    """Cria e devolve o diretório de uma nova versão."""
    root = root or settings.models_root
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    version_dir = root / version
    version_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Nova versão de modelo: %s", version)
    return version_dir


def list_versions(root: Path | None = None) -> list[str]:
    """Lista as versões existentes, da mais antiga para a mais recente."""
    root = root or settings.models_root
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / METADATA_FILENAME).exists())


def write_metadata(version_dir: Path, metadata: dict) -> Path:
    """Grava o ``metadata.json`` da versão."""
    path = version_dir / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_metadata(version_dir: Path) -> dict:
    """Lê o ``metadata.json`` de uma versão."""
    return json.loads((version_dir / METADATA_FILENAME).read_text(encoding="utf-8"))


def promote(version_dir: Path, root: Path | None = None) -> Path:
    """Aponta ``current.json`` para ``version_dir``."""
    root = root or settings.models_root
    root.mkdir(parents=True, exist_ok=True)

    pointer = root / POINTER_FILENAME
    pointer.write_text(
        json.dumps(
            {"version": version_dir.name, "promoted_at": datetime.now(UTC).isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Versão promovida para produção: %s", version_dir.name)
    return pointer


def resolve_current(root: Path | None = None) -> Path | None:
    """Devolve o diretório da versão promovida, ou ``None`` se não houver.

    Se o ponteiro estiver ausente mas existirem versões em disco, cai para a
    mais recente — evita que um clone limpo suba a API sem modelo depois de um
    treino local.
    """
    root = root or settings.models_root
    pointer = root / POINTER_FILENAME

    if pointer.exists():
        version = json.loads(pointer.read_text(encoding="utf-8")).get("version")
        candidate = root / str(version)
        if (candidate / METADATA_FILENAME).exists():
            return candidate
        logger.warning("Ponteiro aponta para versão inexistente: %s", version)

    versions = list_versions(root)
    if not versions:
        return None

    logger.warning("Sem ponteiro válido; usando a versão mais recente: %s", versions[-1])
    return root / versions[-1]
