import pytest
import structlog

from app.utils.logger import setup_structlog


@pytest.fixture(autouse=True)
def reset_structlog_globals():
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def test_setup_structlog_configures_correctly():
    setup_structlog()

    config = structlog.get_config()
    processors = config["processors"]

    assert config["cache_logger_on_first_use"] is True
    assert isinstance(config["logger_factory"], structlog.PrintLoggerFactory)

    assert structlog.contextvars.merge_contextvars in processors
    assert structlog.processors.add_log_level in processors

    processor_types = [type(p) for p in processors]
    assert structlog.processors.TimeStamper in processor_types
