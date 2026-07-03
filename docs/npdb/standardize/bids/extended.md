
# `npdb standardize bids`

- [`npdb standardize bids`](#npdb-standardize-bids)
  - [Association and standard customization](#association-and-standard-customization)
    - [Phenotype dictionary (`--phenotype-dict`)](#phenotype-dictionary---phenotype-dict)
      - [Mappings](#mappings)
      - [Context and onthologies](#context-and-onthologies)
      - [Categorical variable definition](#categorical-variable-definition)
    - [TSV header mappings (\`--)](#tsv-header-mappings---)
  - [Standardization `modes`](#standardization-modes)
    - [`--mode manual` (default)](#--mode-manual-default)
    - [`--mode auto`](#--mode-auto)
    - [`--mode full-auto`](#--mode-full-auto)

## Association and standard customization

### Phenotype dictionary (`--phenotype-dict`)

The phenotype dictionary defines matches between wanted output phenotype descriptors (`age` versus `age_years`, `participant_id` versus `subject_id`, etc.) and their importance (called `confidence`) in the standardization process.

```json
{
   "mappings": { ... },
   "@context": { ... }
}
```

#### Mappings

Each entry in the `mappings` map defines a match between a **wanted phenotype descriptor** and its **input variants** or **aliases**.

```json
{
   "mappings": {
      "wanted_descriptor": {
         "aliases": [ ... ],
         "confidence": ...,
         "variable": "...",
         "format": "...",
         "variableType": "...",
         "levels": { ... },
         "note": "..."
      },
      ...
   }
}
```

|Field|Description|Required|
|-|-|-|
| `wanted_descriptor` | The name of the phenotype descriptor to appear in the output `participants.tsv` file. | true |
| `aliases` | A list of input variants to consider for the `wanted_descriptor`. | true |
| `confidence` | A number between 0 and 1, representing the importance of the `wanted_descriptor` in the standardization process (more than one descriptor can have the same aliases). | true |
| `variable` | The onthology variable name associated with the `wanted_descriptor`. See the [context and onthologies](#context-and-onthologies) section below. | false |
| `format` | The format of the `wanted_descriptor` values. Must be an onthology-compliant format. See the [context and onthologies](#context-and-onthologies) section below. | false |
| `variableType` | The type of the `wanted_descriptor` values. Either an `Identifier`, a `Continuous` or a `Categorical` variable. | false |
| `levels` | A map of the possible values for a `Categorical` variable. See the [categorical variable definition](#categorical-variable-definition) section below. | false |
| `note` | A note describing the `wanted_descriptor`, for documentation purpose only (won't be used in the standardization process). | false |

Below is an example for the `participant_id` standard descriptor, with variants commonly used in the neuroimaging field. Its confidence is maximal (`1.0`) to ensure its unequivocal use in the standardization process :

```json
{
   "participant_id": {
      "aliases": [
         "participant-id",
         "subject", "subject_id", "subject-id",
         "sub", "sub_id", "sub-id",
         "subj", "subj_id", "subj-id",
         "subid", "pid", "id", "identifier"
      ],
      "confidence": 1.0,
      "variable": "nb:ParticipantID",
      "variableType": "Identifier",
      "note": "Unique participant identifier."
   }
}
```

#### Context and onthologies

Onthologies describe understandable, organized, fully described, machine-readable and interpretable **terms** and **concepts**. In the phenotypes dictionaries, they are used to ensure that equivalent columns remain unique (such as `age` and `age_years`), and that the values of categorical variables are consistent (such as `male` and `M`).

The onthologies allowed in a dictionary are dictated by the `@context` field, which maps usable prefixes to their corresponding onthology URLs. We recommend the following onthologies, which covers the neuroimaging field quite well :

```json
{
   "@context": {
      "nb": "http://neurobagel.org/vocab/",
      "ncit": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
      "nidm": "http://purl.org/nidash/nidm#",
      "snomed": "http://purl.bioontology.org/ontology/SNOMEDCT/"
   }
}
```

Explore the links below for terms and concepts available in each onthology :

|||
|-|-|
| NeuroBagel | https://github.com/neurobagel/communities/blob/main/configs/Neurobagel/config.json |
| NCI Thesaurus | https://evsexplore.semantics.cancer.gov/evsexplore/welcome |
| NIDM | https://github.com/incf-nidash/nidm-specs |
| SNOMED CT | https://snomedbrowser.org |

#### Categorical variable definition

### TSV header mappings (`--)

## Standardization `modes`

>[!IMPORTANT]
>Automated modes use the NeuroBagel annotation standard (a minor extension to the BIDS standard) to uniformize the [participants phenotypes](https://neurobagel.org/user_guide/data_prep/) files.

### `--mode manual` (default)

This mode only relies on deterministic associations of vendored or user provided [**phenotype dictionary**](#phenotype-dictionary) and [**TSV header mappings**](#tsv-header-mappings) to standardize the files. Columns and fields that cannot be standardized will be kept left untouched and will be considered as distinct or unique descriptors.

### `--mode auto`

In this mode, the command uses fuzzy logic to match mappings and keeps only matches respecting a given threshold (usually `0.7`) when standardizing `participants.tsv` and generating `participants.json`.

### `--mode full-auto`

In this mode, the command runs the same pipeline as `--mode auto` with a more lenient confidence threshold, accepting more non-static matches.

>[!IMPORTANT]
> This mode will generate **provenance files**, containing important information on the choices made by the language models when editing the metadata files. Inspect them after the command has finished running to ensure that the generated metadata files are correct.
