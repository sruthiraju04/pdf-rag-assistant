import config


def test_is_configured_accepts_runtime_key():
    assert config.is_configured("test-key")


def test_is_configured_rejects_blank_runtime_key(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", None)
    assert not config.is_configured("   ")
