"""
Facade for the gitea2bagel dataset conversion workflow.

DatasetConversionFacade encapsulates the five sequential steps
(clone → extend description → annotate → convert to JSON-LD) so that
cli.py contains only argument parsing, validation, and facade invocation.
"""

import asyncio
import os
import tempfile
import typer

from pathlib import Path

from npdb.annotation import AnnotationConfig
from npdb.managers import DataNeuroPolyMTL, BagelNeuroPolyMTL
from npdb.managers.neurobagel import NeurobagelAnnotator


class DatasetConversionFacade:
    """
    Orchestrates the full gitea-to-Neurobagel conversion pipeline.

    Accepts already-constructed domain objects so that the CLI keeps
    responsibility for credential-loading and config construction.
    """

    def __init__(
        self,
        gitea_manager: DataNeuroPolyMTL,
        annotation_config: AnnotationConfig,
    ) -> None:
        self._gitea = gitea_manager
        self._annotation_config = annotation_config

    async def run(self, dataset: str, output: Path) -> None:
        """
        Execute the full conversion pipeline for *dataset*.

        Steps:
        1. Surface-clone the repository from Gitea
        2. Extend the dataset_description.json with repository metadata
        3. Annotate participants.tsv with the configured annotation mode
        4. Convert the annotated artefacts to Neurobagel JSON-LD

        Args:
            dataset: Repository name in the datasets organisation on Gitea.
            output:  Writable output directory (already validated by caller).

        Raises:
            typer.Exit: on unrecoverable errors (missing participants.tsv,
                        annotation failure without fallback).
        """
        with tempfile.TemporaryDirectory() as local_clone:
            # Step 1: Clone
            self._gitea.clone_repository(dataset, local_clone, light=True)

            # Step 2: Extend description
            dataset_description = self._gitea.extend_description(
                dataset, local_clone)

            # Step 3: Locate participants.tsv
            participants_tsv = os.path.join(local_clone, "participants.tsv")
            if not os.path.exists(participants_tsv):
                typer.echo(
                    "Error: participants.tsv not found in dataset.", err=True)
                raise typer.Exit(code=1)

            # Step 4: Annotate
            mode = self._annotation_config.mode

            if mode == "full-auto":
                typer.echo(
                    "\n⚠️  WARNING: EXPERIMENTAL/UNSTABLE MODE\n"
                    "Full-auto annotation uses AI and browser automation "
                    "without validation.\n"
                    "Review phenotypes_provenance.json before using annotations.\n",
                    err=True,
                )

            annotator = NeurobagelAnnotator(self._annotation_config)
            try:
                success = await annotator.execute(
                    participants_tsv_path=Path(participants_tsv),
                    output_dir=output,
                )
                if not success:
                    typer.echo(
                        f"⚠️  Annotation mode '{mode}' execution failed.",
                        err=True,
                    )
                    if mode == "manual":
                        typer.prompt(
                            "Press Enter once you have saved the phenotypes "
                            "files to continue..."
                        )
            except Exception as e:
                typer.echo(f"Error during annotation: {e}", err=True)
                typer.echo("Falling back to manual annotation.", err=True)
                typer.prompt(
                    "Press Enter once you have saved the phenotypes files to continue..."
                )

            # Step 5: Convert to JSON-LD
            bagel_manager = BagelNeuroPolyMTL(output.absolute().as_posix())
            bagel_manager.convert_bids(
                dataset=dataset,
                bids_dir=local_clone,
                phenotypes_tsv=os.path.join(output, "phenotypes.tsv"),
                phenotypes_annotations=os.path.join(
                    output, "phenotypes_annotations.json"),
                dataset_description=dataset_description,
            )

        typer.echo(f"✅ Conversion complete! Output saved to: {output}")
