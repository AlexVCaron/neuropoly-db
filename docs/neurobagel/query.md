# NeuroBagel querying

## Query User Interface

NeuroBagel exposes a user-friendly web interface to query and explore the datasets available in the node. Use the URL `http://localhost:9000` to access the interface (replace the port number with the value of `NB_QUERY_PORT_HOST` if you changed it in the `.env` file).

Once open, **click on `Submit Query`** to display the full list of available datasets.

![NeuroBagel query interface](../assets/neurobagel_query_all.png)

### Refining the query results

Use the **filters stacked on the left of the interface** to refine the query results using _age ranges_, _sex_, _diagnoses_, _imaging modalities_ and more. Once satisfied, **click on the `Submit Query`** button again.

Datasets matching the selected filters will be displayed on the right, with the number of matching subjects they contain, as well as the list of available data modalities associated to their imaging sessions.

![NeuroBagel query interface with filters](../assets/neurobagel_query_subset.png)

### Exporting the query results

Use the **tick boxes on the left of each dataset card** to select the datasets to export. This activates the **Download** button at the bottom right of the interface.

![NeuroBagel query interface with filters](../assets/neurobagel_query_export.png)

The exported query results is saved in a **T**ab-**S**eparated-**V**alue (**TSV**) file, with one line per subject and imaging/phenotypic session. In it you'll find most of the metadata associated with the subjects matching the query, like their age, sex, diagnosis, etc. The table below describes some interesting columns, aside `sex`, `age` and `diagnosis` :

|                              |                                                                                                                    |
|------------------------------|--------------------------------------------------------------------------------------------------------------------|
|        `DatasetName`         | Name of the dataset the subject belongs to.                                                                        |
|       `RepositoryURL`        | URL of the repository hosting the dataset.                                                                         |
|         `SubjectID`          | Identifier of the subject.                                                                                         |
|         `SessionID`          | Identifier of the imaging/phenotypic session.                                                                      |
|    `ImagingSessionPath`      | Relative path to the imaging session in the repository.                                                            |
| `SessionImagingModalities`   | Name of the imaging modalities available in the session (e.g. `T1w`, `T2w`, `fMRI`, etc.).                        |
| `SessionCompletedPipelines`  | Comma-separated list of derivative pipelines completed for this session (e.g. `fmriprep,mriqc`). Empty when no pipeline filter was applied in the query. |
|        `AccessLink`          | Link to access the session data.                                                                                    |

### Downloading the imaging data from the query results

> [!TIP]
> First, install the `npdb` command line tool, following the [instructions here](../npdb/install.md).

The exported query results contains an `AccessLink` column that, when possible, will be filled with an URL to download the imaging data associated with each session. **For datasets indexed on `git`, this is not possible.** Instead, use the `npdb download` command line tool with the `--git` option (additionally use the `--git-annex` option if necessary) :

```bash
uv run npdb download --git --git-annex <query-results.tsv>
```

`npdb download` automatically:

- **Skips phenotypic rows** — rows with no imaging path are silently filtered and a count is reported.
- **Downloads pipeline derivatives** — when the query included a pipeline filter, `SessionCompletedPipelines` is populated and `npdb download` will additionally fetch `derivatives/<pipeline>/sub-*` for each matching subject.

#### SSL certificate errors

If the Gitea server uses a self-signed certificate you may see an error like:

```
SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate
```

Pass `--no-verify-ssl` to skip certificate verification:

```bash
uv run npdb download --git --git-annex --no-verify-ssl <query-results.tsv>
```

> [!WARNING]
> Only use `--no-verify-ssl` on trusted private networks. Disabling certificate verification exposes you to man-in-the-middle attacks.

#### Creating a Gitea access token

The `--git` mode requires three environment variables: `NP_GITEA_APP_URL`, `NP_GITEA_APP_USER`, and `NP_GITEA_APP_TOKEN`. See the [Gitea token guide](./gitea_token.md) for step-by-step instructions on how to create a personal access token.
