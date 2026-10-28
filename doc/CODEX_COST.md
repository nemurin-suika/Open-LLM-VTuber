# Codex monthly cost display

The chat sidebar displays this calendar month's estimated cost when the configured
Letta model uses the `codex_cli/` provider. The bar stays outside the scrolling
message list. It refreshes every 30 seconds and when the browser tab becomes visible.

This integration requires the organization's approved **codex-usage 0.1.3**
distribution, whose description is "Local Codex token usage and cost reports by
model". A different public package uses the same name and higher version numbers.
Do not install or upgrade by name alone. Install the verified local distribution
into the OLV virtual environment with `uv pip install --python .venv/bin/python
--no-deps /path/to/approved/codex_usage-0.1.3.tar.gz`. Private registry addresses and
credentials belong only in local package-manager configuration, never this repository.
The optional package is deliberately not added to public project dependencies.

The adapter uses `codex_usage.cli.build_report` and `parse_args` from that exact
version. Revalidate the adapter before changing the package version. No pricing
table, provider identifiers, account information, or package source is copied into
this repository or returned by the endpoint.

`GET /codex-usage` returns the amount, currency, inclusive month boundaries in KST,
last successful collection time, and completeness indicators. Amounts are decimal
strings. The result covers all local records, not just the active chat or OLV.
It is an estimate using the package's bundled prices, not an invoice. Deleted logs,
other devices, tax, and discounts cannot be reconstructed by this integration.
Partial totals are labeled; unavailable data is never shown as a zero charge.

The server runs collection in a subprocess with a 15-second timeout and shares a
30-second cache across clients. A failed refresh retains a visibly stale previous
value within the same month, but never carries it into a new month.

The log home defaults to `CODEX_HOME` or the real user's `~/.codex`; override it
with `VTUBER_CODEX_USAGE_HOME` in the service's ignored local environment. The
working directory for the package's additional `./usage` scan is
`VTUBER_CODEX_EXEC_ROOT`, then `MIZUKI_AI_HOME`, then the real user's `~/ai`.
Both services must agree on these locations. `MIZUKI_REAL_HOME` overrides the home.

Letta's ephemeral Codex provider writes request usage records into the persistent
Codex home's `usage/YYYY/MM/DD` directory. These contain only identifiers, model,
provider, completion time, token counts, and completeness, never prompts or answers.
Only reported tokens are persisted; the response-length fallback is not used for
cost estimation. Interrupted calls without usage are marked unavailable. Records
are private local files and are not committed. Past ephemeral calls made before
this change have no persisted usage and are not included in the monthly estimate.

Build the web repository with `npm run build:web`. Restart OLV to register the new
endpoint and Letta to activate usage recording, coordinating with ongoing sessions.
Reload the browser after deployment. Building or saving files alone does not update
the running Python services.
