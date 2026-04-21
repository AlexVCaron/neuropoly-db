
import asyncio
import json
import os
import shutil

from pathlib import Path
from typer.testing import CliRunner
from bagel.cli import bagel

from npdb.annotation import AnnotationConfig
from npdb.annotation.duplicates import resolve_phenotype_duplicates
from npdb.annotation.provenance import ProvenanceReport, add_column_provenance
from npdb.annotation.standardize import apply_header_map, load_header_map
from npdb.annotation.strategies import AnnotationStrategyFactory, AnnotatorContext
from npdb.automation.mappings.resolvers import MappingResolver
from npdb.external.neurobagel.automation import NBAnnotationToolBrowserSession
from npdb.external.neurobagel.schema import convert_to_bagel_schema
from npdb.managers.annotation import AnnotationManager
from npdb.managers.model import Manager
from npdb.utils import parse_tsv_columns


class BagelDB:
    def __init__(self, jsonld_root: str):
        self.root = jsonld_root


class BagelMixin:
    def __init__(self, db: BagelDB):
        self.cli = CliRunner()
        self.db = db

    def bids2tsv(self, bids_directory: str, output_tsv: str):
        self._run_bagel_cli(
            "bids2tsv",
            "--bids-dir", bids_directory,
            "--output", output_tsv,
            "--overwrite"
        )

    def bagel_pheno(
        self,
        dataset_name: str,
        phenotypes_tsv: str,
        phenotypes_annotations: str,
        dataset_description: str
    ):
        self._run_bagel_cli(
            "pheno",
            "--pheno", phenotypes_tsv,
            "--dictionary", phenotypes_annotations,
            "--dataset-description", dataset_description,
            "--output", os.path.join(self.db.root, f"{dataset_name}.jsonld"),
            "--overwrite"
        )

    def bagel_bids(
        self,
        dataset_name: str,
        bids_table: str
    ):
        jsonld_path = os.path.join(self.db.root, f"{dataset_name}.jsonld")
        self._run_bagel_cli(
            "bids",
            "--jsonld-path", jsonld_path,
            "--bids-table", bids_table,
            "--output", jsonld_path,
            "--overwrite"
        )

    def _run_bagel_cli(self, *args):
        result = self.cli.invoke(bagel, args)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Bagel CLI failed with exit code {result.exit_code} and output: {result.output}")


class NeurobagelManager(Manager):
    def __init__(self, jsonld: str):
        self.db = BagelDB(jsonld)

    @property
    def datasets(self):
        return os.listdir(self.db.root)


class NeurobagelAnnotator(AnnotationManager):
    """
    Orchestrates phenotype annotation automation across 4 modes.

    Integrates:
    - Browser session management (Playwright)
    - Mapping resolution (static dict + fuzzy matching)
    - Provenance tracking (audit trail)
    - Mode-specific orchestration (manual/assist/auto/full-auto)
    """

    def __init__(self, config: AnnotationConfig):
        super().__init__(config)

    async def _save_outputs(
        self,
        participants_tsv_path: Path,
        output_dir: Path,
        annotations_dict: dict
    ) -> None:
        """
        Save phenotypes.tsv and phenotypes_annotations.json to output directory.

        Processing pipeline:
        1. Save flat-format annotations and TSV
        2. Apply duplicate resolver (modifies TSV + JSON in-place)
        3. Load resolved flat format and convert to Bagel schema
        4. Save Bagel-compliant format

        Args:
            participants_tsv_path: Path to input participants.tsv
            output_dir: Output directory where files should be saved
            annotations_dict: Mapping annotations as dictionary (flat format)
        """
        # Copy participants.tsv as phenotypes.tsv
        phenotypes_tsv_path = output_dir / "phenotypes.tsv"
        shutil.copy2(participants_tsv_path, phenotypes_tsv_path)
        print(f"✓ Saved phenotypes.tsv: {phenotypes_tsv_path}")

        # Step 1: Save flat-format annotations to JSON
        phenotypes_annotations_path = output_dir / "phenotypes_annotations.json"
        with open(phenotypes_annotations_path, 'w') as f:
            json.dump(annotations_dict, f, indent=2)
        print(
            f"✓ Saved flat-format annotations: {phenotypes_annotations_path}")

        # Step 2: Apply duplicate resolver (modifies both files in-place)
        # This resolves duplicates in both TSV columns and JSON annotations
        print(f"→ Resolving duplicate field mappings...")
        resolve_phenotype_duplicates(
            phenotypes_tsv_path,
            phenotypes_annotations_path,
            verbose=True
        )

        # Step 3: Load resolved flat-format annotations
        with open(phenotypes_annotations_path, 'r') as f:
            resolved_annotations_flat = json.load(f)

        # Step 4: Convert resolved annotations to Bagel schema
        # Note: self.resolver.mappings is the full phenotype_mappings structure with @context and mappings keys
        phenotype_mappings_dict = self.resolver.mappings
        resolved_annotations_bagel = convert_to_bagel_schema(
            resolved_annotations_flat,
            phenotype_mappings_dict
        )

        # Step 5: Save Bagel-compliant format (overwrites flat format)
        with open(phenotypes_annotations_path, 'w') as f:
            json.dump(resolved_annotations_bagel, f, indent=2)
        print(
            f"✓ Saved Bagel-compliant annotations: {phenotypes_annotations_path}")

    async def execute(
        self,
        participants_tsv_path: Path,
        output_dir: Path,
    ) -> bool:
        """
        Execute annotation automation according to configured mode.

        Args:
            participants_tsv_path: Path to participants.tsv file
            output_dir: Output directory for phenotypes files

        Returns:
            True if successful, False on failure
        """
        if not participants_tsv_path.exists():
            raise FileNotFoundError(
                f"Participants TSV not found: {participants_tsv_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Apply user-supplied header translation map before any annotation mode
        if self.config.header_map:
            hmap = load_header_map(self.config.header_map)
            pre_renames = apply_header_map(participants_tsv_path, hmap)
            if pre_renames:
                print(
                    f"✓ Header map applied: renamed {len(pre_renames)} columns")
                for old, new in pre_renames.items():
                    print(f"  {old} → {new}")

        ctx = AnnotatorContext(
            config=self.config,
            resolver=self.resolver,
            provenance=self.provenance,
            save_outputs=self._save_outputs,
        )
        strategy = AnnotationStrategyFactory.create(self.config)
        return await strategy.run(participants_tsv_path, output_dir, ctx)
