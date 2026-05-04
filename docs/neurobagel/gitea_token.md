# Creating a Gitea personal access token

The `npdb download --git` command authenticates against the Gitea server using a **personal access token** stored in the `NP_GITEA_APP_TOKEN` environment variable.  The steps below show how to create one.

## Steps

1. **Log in** to the Gitea instance (e.g. `https://data.neuro.polymtl.ca`) with your account credentials.

2. Open your **user settings** by clicking your avatar in the top-right corner and selecting **Settings**.

   ![Gitea user menu](../assets/gitea_token_step1.png)

3. In the left sidebar choose **Applications**.

   ![Applications menu item](../assets/gitea_token_step2.png)

4. Under **Manage Access Tokens**, fill in a **Token Name** (e.g. `npdb-download`) and choose the minimum required scopes:

   | Scope | Permission |
   |-------|------------|
   | `repository` | Read |

   Click **Generate Token**.

   ![Token generation form](../assets/gitea_token_step3.png)

5. **Copy the token immediately** — it will only be shown once.

   ![Copy token](../assets/gitea_token_step4.png)

6. Add the token to your environment (or to the `.env` file at the root of the repository):

   ```bash
   export NP_GITEA_APP_URL=https://data.neuro.polymtl.ca
   export NP_GITEA_APP_USER=<your-gitea-username>
   export NP_GITEA_APP_TOKEN=<paste-token-here>
   ```

   Or in `.env`:

   ```dotenv
   NP_GITEA_APP_URL=https://data.neuro.polymtl.ca
   NP_GITEA_APP_USER=<your-gitea-username>
   NP_GITEA_APP_TOKEN=<paste-token-here>
   ```

> [!CAUTION]
> Never commit your token to a public repository.  Add `.env` to `.gitignore` if it is not already there.
