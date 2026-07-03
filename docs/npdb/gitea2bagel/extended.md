# `npdb gitea2bagel`

- [`npdb gitea2bagel`](#npdb-gitea2bagel)
  - [Annotation and standardization `modes`](#annotation-and-standardization-modes)
    - [`--mode manual` (default)](#--mode-manual-default)
    - [`--mode assist`](#--mode-assist)
    - [`--mode auto`](#--mode-auto)
    - [`--mode full-auto`](#--mode-full-auto)

## Annotation and standardization `modes`

### `--mode manual` (default)

To serve datasets to **NeuroBagel**, they first need to be annotated and standardized. via a [**manual procedure described on the NeuroBagel website**](https://neurobagel.org/user_guide/dataset_description/).

When running `npdb gitea2bagel` in this mode, you will be prompted to provide the required files and informations at the right locations, in time.

### `--mode assist`

In this mode, the command will use [`playwright`](https://playwright.dev/) to automate interations with the [NeuroBagel annotation web application](https://annotate.neurobagel.org), but will ask you to **confirm actions taken in the browser and their results**. To use it, first prepare your dataset(s) :

1. [Format the `dataset_description.json` file](https://neurobagel.org/user_guide/dataset_description/)

2. [Format the `participants.tsv` file](https://neurobagel.org/user_guide/data_prep/)

   The `npdb` project extends NeuroBagel with **additional pathologies and diseases**. Look at `config/categorical_terms.json`, for a complete list of `terms` and their `aliases` supported by the command.

3. [Validate the imaging data structure](https://neurobagel.org/user_guide/preparing_imaging_data/)

   The `npdb` project extends NeuroBagel with **additional modalities and imaging techniques**. Look at `config/neuropoly_imaging_modalities.json` and `config/imaging_extensions.json` to see the full list of supported modalities and extensions.

### `--mode auto`

In this mode runs the **same automations as in `--mode assist`**, but also uses fuzzy logic to propose editions to the `dataset_description.json` and `participants.tsv` files to align them to the NeuroBagel standard. The user will be prompted to **confirm or reject each proposed edition**.

>[!IMPORTANT]
> Review the [NeuroBagel manual installation procedure](https://neurobagel.org/user_guide/dataset_description/) to **ensure that all required metadata fields are present in your files before running this mode.**

### `--mode full-auto`

In this mode, the command will run the **same automations as in `--mode auto`**, but will **not ask for any user confirmation**. The command will automatically edit the `dataset_description.json` and `participants.tsv` files to align them to the NeuroBagel standard.

>[!IMPORTANT]
> This mode will generate **provenance files**, containing important information on the choices made by the language models when editing the metadata files. Inspect them after the command has finished running to ensure that the generated metadata files are correct.
