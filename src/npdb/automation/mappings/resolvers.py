"""
Mapping resolver for precedence-based column-to-variable mapping.

Chains three layers of authority:
1. Static dictionary (highest confidence, repo-maintained)
2. Deterministic fuzzy matching (medium confidence, rule-based)
3. AI suggestions (lowest confidence, optional, deferred)

Resolver returns per-column mapping with source and confidence tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from npdb.automation.mappings.solvers import load_static_mappings, load_user_mappings, merge_mappings
from npdb.annotation.matching import ColumnMatcher


@dataclass
class ResolvedMapping:
    """Result of resolving a column header to a phenotype variable."""
    column_name: str
    mapped_variable: str
    confidence: float
    source: str  # "static", "deterministic", "ai", or "unresolved"
    mapping_data: Dict[str, Any]
    rationale: str


# ---------------------------------------------------------------------------
# Chain-of-Responsibility handlers
# ---------------------------------------------------------------------------

class ResolutionHandler(ABC):
    """Abstract node in the column-resolution chain."""

    def __init__(self) -> None:
        self._next: Optional[ResolutionHandler] = None

    def set_next(self, handler: "ResolutionHandler") -> "ResolutionHandler":
        """Attach *handler* as the next node and return it (fluent API)."""
        self._next = handler
        return handler

    @abstractmethod
    def handle(
        self, column_name: str, matcher: ColumnMatcher
    ) -> Optional[ResolvedMapping]:
        """
        Try to resolve *column_name*.

        Return a ResolvedMapping on success, or delegate to the next
        handler (returning None if the chain is exhausted).
        """
        ...


class StaticResolutionHandler(ResolutionHandler):
    """Resolves columns via exact key lookup in the static dictionary."""

    def handle(
        self, column_name: str, matcher: ColumnMatcher
    ) -> Optional[ResolvedMapping]:
        mapping_data = matcher.get_mapping_data(column_name)
        if mapping_data:
            return ResolvedMapping(
                column_name=column_name,
                mapped_variable=mapping_data.get("variable", "unknown"),
                confidence=mapping_data.get("confidence", 0.95),
                source="static",
                mapping_data=mapping_data,
                rationale="Exact match in static dictionary",
            )
        if self._next:
            return self._next.handle(column_name, matcher)
        return None


class FuzzyResolutionHandler(ResolutionHandler):
    """Resolves columns via deterministic fuzzy matching against dict aliases."""

    def __init__(
        self,
        exact_threshold: float = 1.0,
        fuzzy_threshold: float = 0.75,
    ) -> None:
        super().__init__()
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold

    def handle(
        self, column_name: str, matcher: ColumnMatcher
    ) -> Optional[ResolvedMapping]:
        match_result = matcher.match_column(
            column_name,
            exact_threshold=self.exact_threshold,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        if match_result:
            mapping_key, confidence, match_source = match_result
            mapping_data = matcher.get_mapping_data(mapping_key)
            if mapping_data:
                return ResolvedMapping(
                    column_name=column_name,
                    mapped_variable=mapping_data.get("variable", "unknown"),
                    confidence=confidence,
                    source="deterministic",
                    mapping_data=mapping_data,
                    rationale=(
                        f"Fuzzy match: '{column_name}' → '{mapping_key}' "
                        f"({match_source}, score {confidence:.2f})"
                    ),
                )
        if self._next:
            return self._next.handle(column_name, matcher)
        return None


class AIResolutionHandler(ResolutionHandler):
    """
    Stub for future AI-assisted resolution.

    Currently always returns None (passes through the chain).
    Wire in an AI provider here when the feature is enabled.
    """

    def handle(
        self, column_name: str, matcher: ColumnMatcher
    ) -> Optional[ResolvedMapping]:
        if self._next:
            return self._next.handle(column_name, matcher)
        return None


class MappingResolver:
    """
    Resolves column headers to Neurobagel standardized variables via precedence chain.

    Precedence order:
    1. Static dictionary (user-supplied or built-in)
    2. Fuzzy matching against static dict keys/aliases
    3. AI suggestions (deferred; stub returns None today)
    """

    def __init__(
        self,
        user_dictionary_path: Optional[str] = None,
        exact_threshold: float = 1.0,
        fuzzy_threshold: float = 0.75
    ):
        """
        Initialize resolver with optional user dictionary override.

        Args:
            user_dictionary_path: Optional path to user-supplied phenotype_mappings.json.
            exact_threshold: Confidence threshold for exact matching (default 1.0).
            fuzzy_threshold: Confidence threshold for fuzzy matching (default 0.75).
        """
        # Load and merge mappings
        static_mappings = load_static_mappings()
        if user_dictionary_path:
            user_mappings = load_user_mappings(user_dictionary_path)
            self.mappings = merge_mappings(static_mappings, user_mappings)
        else:
            self.mappings = static_mappings

        # Initialize fuzzy matcher
        self.matcher = ColumnMatcher(self.mappings)

        # Thresholds for matching
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold

        # Cache resolved mappings (_resolved_cache lives here, not on handlers)
        self._resolved_cache: Dict[str, ResolvedMapping] = {}

        # Build resolution chain: static → fuzzy → ai_stub
        _static = StaticResolutionHandler()
        _fuzzy = FuzzyResolutionHandler(
            exact_threshold=self.exact_threshold,
            fuzzy_threshold=self.fuzzy_threshold,
        )
        _ai = AIResolutionHandler()
        _static.set_next(_fuzzy).set_next(_ai)
        self._resolution_chain: ResolutionHandler = _static

    def resolve_column(self, column_name: str) -> ResolvedMapping:
        """
        Resolve a column header to a phenotype variable via the handler chain.

        Attempts in order: static dictionary → fuzzy matching → AI stub.
        Falls back to an "unresolved" mapping if the chain returns None.

        Args:
            column_name: Column header from dataset.

        Returns:
            ResolvedMapping with source, confidence, and mapping data.
        """
        if column_name in self._resolved_cache:
            return self._resolved_cache[column_name]

        resolved = self._resolution_chain.handle(column_name, self.matcher)

        if resolved is None:
            resolved = ResolvedMapping(
                column_name=column_name,
                mapped_variable="",
                confidence=0.0,
                source="unresolved",
                mapping_data={},
                rationale=(
                    f"No static or fuzzy match found for '{column_name}'; "
                    f"requires AI suggestion or manual annotation"
                ),
            )

        self._resolved_cache[column_name] = resolved
        return resolved

    def resolve_columns(self, column_names: List[str]) -> List[ResolvedMapping]:
        """
        Resolve multiple column headers in batch.

        Args:
            column_names: List of column headers from dataset.

        Returns:
            List of ResolvedMapping for each column.
        """
        return [self.resolve_column(name) for name in column_names]

    def get_resolution_summary(self, resolved_mappings: List[ResolvedMapping]) -> Dict[str, Any]:
        """
        Generate summary statistics on resolution quality.

        Args:
            resolved_mappings: List of ResolvedMapping results.

        Returns:
            Summary dict with source counts, confidence distribution, unresolved list.
        """
        source_counts = {"static": 0,
                         "deterministic": 0, "ai": 0, "unresolved": 0}
        confidence_scores = []
        unresolved = []

        for mapping in resolved_mappings:
            source_counts[mapping.source] += 1
            if mapping.source != "unresolved":
                confidence_scores.append(mapping.confidence)
            else:
                unresolved.append(mapping.column_name)

        # Compute confidence distribution
        confidence_dist = {
            "high": sum(1 for s in confidence_scores if s >= 0.85),
            "medium": sum(1 for s in confidence_scores if 0.7 <= s < 0.85),
            "low": sum(1 for s in confidence_scores if 0.5 <= s < 0.7),
            "unresolved": len(unresolved)
        }

        return {
            "source_counts": source_counts,
            "confidence_distribution": confidence_dist,
            "unresolved_columns": unresolved,
            "total_resolved": len(resolved_mappings) - len(unresolved),
            "total_columns": len(resolved_mappings)
        }

    def clear_cache(self) -> None:
        """Clear the resolution cache (for testing or fresh resolution)."""
        self._resolved_cache.clear()
