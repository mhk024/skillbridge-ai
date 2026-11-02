from src.utils.config import get_settings


def test_llm_not_configured_when_key_empty(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    assert get_settings().llm_configured is False


def test_llm_configured_when_key_present(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "dummy-test-value")
    assert get_settings().llm_configured is True


def test_api_key_not_in_repr(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    assert "super-secret-value" not in repr(get_settings())


def test_code_execution_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_CODE_EXECUTION", "false")
    assert get_settings().enable_code_execution is False