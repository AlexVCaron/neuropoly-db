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

### Downloading the imaging data from the query results

> [!TIP]
> First, install the `npdb` command line tool, following the [instructions here](../npdb/install.md).

The exported query results may contain an `AccessLink` column that, for some datasets, will be filled with a URL to download the imaging data associated with each session. **For datasets indexed on `git`, this is not possible.** Instead, use the `npdb download` command line tool with the `--git` option (additionally use the `--git-annex` option if necessary) :

```bash
uv run npdb download --git --git-annex <query-results.tsv>
```

`npdb download` automatically **skips phenotypic rows** — rows with no imaging path are filtered and a count is reported.

#### Downloading pipeline derivatives

Pipeline derivatives (e.g. `fmriprep`, `mriqc`) are **not** downloaded by default.  They are only included when you explicitly opt in:

| How | What is downloaded |
|-----|-------------------|
| `--derivatives fmriprep` | Only `derivatives/fmriprep/sub-*` paths |
| `--derivatives fmriprep --derivatives mriqc` | Only those two pipelines |
| `NPDB_DOWNLOAD_DERIVATIVES=1` (env var) | All pipelines listed in `SessionCompletedPipelines` |

The `--derivatives` flag always has **whole precedence** over `NPDB_DOWNLOAD_DERIVATIVES`.  Only pipelines present in `SessionCompletedPipelines` (i.e. matched by the query) are ever fetched; unknown pipeline names are silently skipped.

```bash
# Download raw data + fmriprep derivatives
uv run npdb download --git --derivatives fmriprep <query-results.tsv>

# Download raw data + all available derivatives (via env var)
NPDB_DOWNLOAD_DERIVATIVES=1 uv run npdb download --git <query-results.tsv>
```

> [!NOTE]
> `SessionCompletedPipelines` is only populated in the TSV when you applied a pipeline filter in the NeuroBagel query UI.  Without that filter the column is empty and no derivatives can be fetched.

#### SSL certificate errors

If the Gitea server uses a self-signed certificate you may see an error like:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate
```

Pass `--no-verify-ssl` to skip certificate verification:

```bash
uv run npdb download --git --git-annex --no-verify-ssl <query-results.tsv>
```

> [!WARNING]
> Only use `--no-verify-ssl` on trusted private networks. Disabling certificate verification exposes you to man-in-the-middle attacks.

#### Creating a Gitea access token

The `--git` mode requires three environment variables: `NP_GITEA_APP_URL`, `NP_GITEA_APP_USER`, and `NP_GITEA_APP_TOKEN`. See the [Gitea token guide](./gitea_token.md) for step-by-step instructions on how to create a personal access token.
