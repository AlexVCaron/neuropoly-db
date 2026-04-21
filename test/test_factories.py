"""
Unit tests for GiteaManagerFactory and AnnotationConfigFactory.

Covers:
- Missing env vars raise ValueError with a descriptive message
- Valid env vars return a correctly configured DataNeuroPolyMTL
- Valid CLI args return a correctly configured AnnotationConfig
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from npdb.factories import GiteaManagerFactory, AnnotationConfigFactory
from npdb.annotation import AnnotationConfig


# ── GiteaManagerFactory ───────────────────────────────────────────────────────


class TestGiteaManagerFactory:
    """Tests for GiteaManagerFactory.create_from_env."""

    _VALID_ENV = {
        "NP_GITEA_APP_URL": "https://gitea.example.com",
        "NP_GITEA_APP_USER": "testuser",
        "NP_GITEA_APP_TOKEN": "abc123token",
    }

    def test_raises_when_all_env_vars_missing(self, monkeypatch):
        """All three required vars missing → ValueError listing all names."""
        for var in ("NP_GITEA_APP_URL", "NP_GITEA_APP_USER", "NP_GITEA_APP_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(ValueError) as exc_info:
            GiteaManagerFactory.create_from_env()

        msg = str(exc_info.value)
        assert "NP_GITEA_APP_URL" in msg
        assert "NP_GITEA_APP_USER" in msg
        assert "NP_GITEA_APP_TOKEN" in msg

    def test_raises_when_url_missing(self, monkeypatch):
        """Missing URL env var → ValueError mentioning that var."""
        monkeypatch.delenv("NP_GITEA_APP_URL", raising=False)
        monkeypatch.setenv("NP_GITEA_APP_USER", "user")
        monkeypatch.setenv("NP_GITEA_APP_TOKEN", "tok")

        with pytest.raises(ValueError, match="NP_GITEA_APP_URL"):
            GiteaManagerFactory.create_from_env()

    def test_raises_when_user_missing(self, monkeypatch):
        """Missing USER env var → ValueError mentioning that var."""
        monkeypatch.setenv("NP_GITEA_APP_URL", "https://gitea.example.com")
        monkeypatch.delenv("NP_GITEA_APP_USER", raising=False)
        monkeypatch.setenv("NP_GITEA_APP_TOKEN", "tok")

        with pytest.raises(ValueError, match="NP_GITEA_APP_USER"):
            GiteaManagerFactory.create_from_env()

    def test_raises_when_token_missing(self, monkeypatch):
        """Missing TOKEN env var → ValueError mentioning that var."""
        monkeypatch.setenv("NP_GITEA_APP_URL", "https://gitea.example.com")
        monkeypatch.setenv("NP_GITEA_APP_USER", "user")
        monkeypatch.delenv("NP_GITEA_APP_TOKEN", raising=False)

        with pytest.raises(ValueError, match="NP_GITEA_APP_TOKEN"):
            GiteaManagerFactory.create_from_env()

    def test_valid_env_returns_manager(self, monkeypatch):
        """All vars present → returns DataNeuroPolyMTL without error."""
        for var, val in self._VALID_ENV.items():
            monkeypatch.setenv(var, val)

        # Patch DataNeuroPolyMTL.__init__ so no real network call is made.
        with patch("npdb.managers.DataNeuroPolyMTL.__init__", return_value=None):
            manager = GiteaManagerFactory.create_from_env(ssl_verify=False)

        from npdb.managers import DataNeuroPolyMTL
        assert isinstance(manager, DataNeuroPolyMTL)

    def test_ssl_verify_forwarded(self, monkeypatch):
        """ssl_verify argument is forwarded to DataNeuroPolyMTL constructor."""
        for var, val in self._VALID_ENV.items():
            monkeypatch.setenv(var, val)

        captured = {}

        def fake_init(self, url, user, token, ssl_verify=True):
            captured["ssl_verify"] = ssl_verify

        with patch("npdb.managers.DataNeuroPolyMTL.__init__", fake_init):
            GiteaManagerFactory.create_from_env(ssl_verify=False)

        assert captured["ssl_verify"] is False


# ── AnnotationConfigFactory ───────────────────────────────────────────────────


class TestAnnotationConfigFactory:
    """Tests for AnnotationConfigFactory.create_from_cli_args."""

    def test_returns_annotation_config(self):
        """Returns an AnnotationConfig instance."""
        config = AnnotationConfigFactory.create_from_cli_args(mode="manual")
        assert isinstance(config, AnnotationConfig)

    def test_mode_is_set(self):
        """mode is passed through unchanged."""
        for mode in ("manual", "assist", "auto", "full-auto"):
            config = AnnotationConfigFactory.create_from_cli_args(mode=mode)
            assert config.mode == mode

    def test_defaults_are_applied(self):
        """Keyword defaults match AnnotationConfig's expected defaults."""
        config = AnnotationConfigFactory.create_from_cli_args(mode="auto")
        assert config.headless is True
        assert config.timeout == 300
        assert config.artifacts_dir is None
        assert config.ai_provider is None
        assert config.ai_model is None
        assert config.phenotype_dictionary is None
        assert config.header_map is None

    def test_optional_fields_forwarded(self, tmp_path):
        """All optional keyword arguments are forwarded to AnnotationConfig."""
        dict_file = tmp_path / "dict.json"
        dict_file.write_text("{}")
        hmap_file = tmp_path / "hmap.json"
        hmap_file.write_text("{}")

        config = AnnotationConfigFactory.create_from_cli_args(
            mode="full-auto",
            headless=False,
            timeout=120,
            artifacts_dir=tmp_path,
            ai_provider="ollama",
            ai_model="neural-chat",
            phenotype_dictionary=dict_file,
            header_map=hmap_file,
        )

        assert config.mode == "full-auto"
        assert config.headless is False
        assert config.timeout == 120
        assert config.artifacts_dir == tmp_path
        assert config.ai_provider == "ollama"
        assert config.ai_model == "neural-chat"
        assert config.phenotype_dictionary == dict_file
        assert config.header_map == hmap_file
