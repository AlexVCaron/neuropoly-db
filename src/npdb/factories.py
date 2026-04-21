"""
Factory Methods for constructing configured domain managers.

Centralises all environment-variable access and cross-field defaults so that
cli.py and facade.py remain free of credential-loading logic.
"""

import os
from pathlib import Path
from typing import Optional

from npdb.annotation import AnnotationConfig
from npdb.managers import DataNeuroPolyMTL


class GiteaManagerFactory:
    """Factory for constructing a :class:`DataNeuroPolyMTL` from environment variables."""

    _URL_VAR = "NP_GITEA_APP_URL"
    _USER_VAR = "NP_GITEA_APP_USER"
    _TOKEN_VAR = "NP_GITEA_APP_TOKEN"

    @classmethod
    def create_from_env(cls, ssl_verify: bool = True) -> DataNeuroPolyMTL:
        """
        Build and return a :class:`DataNeuroPolyMTL` using credentials from the
        environment.

        Args:
            ssl_verify: Whether to verify SSL certificates for Gitea connections.

        Returns:
            A configured :class:`DataNeuroPolyMTL` instance.

        Raises:
            ValueError: If any of the required environment variables
                        (``NP_GITEA_APP_URL``, ``NP_GITEA_APP_USER``,
                        ``NP_GITEA_APP_TOKEN``) are unset or empty.
        """
        url = os.environ.get(cls._URL_VAR, "")
        user = os.environ.get(cls._USER_VAR, "")
        token = os.environ.get(cls._TOKEN_VAR, "")

        missing = [
            var
            for var, val in (
                (cls._URL_VAR, url),
                (cls._USER_VAR, user),
                (cls._TOKEN_VAR, token),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )

        return DataNeuroPolyMTL(url=url, user=user, token=token, ssl_verify=ssl_verify)


class AnnotationConfigFactory:
    """Factory for constructing :class:`AnnotationConfig` from validated CLI arguments."""

    @classmethod
    def create_from_cli_args(
        cls,
        *,
        mode: str,
        headless: bool = True,
        timeout: int = 300,
        artifacts_dir: Optional[Path] = None,
        ai_provider: Optional[str] = None,
        ai_model: Optional[str] = None,
        phenotype_dictionary: Optional[Path] = None,
        header_map: Optional[Path] = None,
    ) -> AnnotationConfig:
        """
        Build and return an :class:`AnnotationConfig` from validated CLI arguments.

        All cross-field rules (e.g. AI provider/model pairing) are expected to have
        been enforced by the caller before invoking this factory.

        Args:
            mode: Annotation mode — one of ``manual``, ``assist``, ``auto``,
                  ``full-auto``.
            headless: Run the automation browser in headless mode.
            timeout: Per-step timeout in seconds.
            artifacts_dir: Optional directory for screenshots/traces.
            ai_provider: AI provider name (e.g. ``"ollama"``).
            ai_model: AI model identifier.
            phenotype_dictionary: Optional path to a phenotype prefill dictionary.
            header_map: Optional path to a header-map JSON file.

        Returns:
            A configured :class:`AnnotationConfig` instance.
        """
        return AnnotationConfig(
            mode=mode,
            headless=headless,
            timeout=timeout,
            artifacts_dir=artifacts_dir,
            ai_provider=ai_provider,
            ai_model=ai_model,
            phenotype_dictionary=phenotype_dictionary,
            header_map=header_map,
        )
