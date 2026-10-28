#!/usr/bin/env python
"""The `wandb` CLI with one upstream bug patched out, in-process.

THE BUG (seen with wandb 0.25.1 against api.wandb.ai, 2026-10-28). Every
`wandb sync` reported success and uploaded nothing. `upsert_run` -- the call
that CREATES the run server-side -- begins by probing the server for a
`ServerSettings` GraphQL type:

    res = self.gql(ProbeServerSettings)
    ... for field in res.get("ServerSettingsType", {}).get("fields", [{}])

`ServerSettings` is a **self-hosted-only** type. The public cloud answers

    {"data": {"ServerSettingsType": null}}

so the key is PRESENT with a null value, the `{}` default never applies, and
`.get("fields")` runs on None: `AttributeError` -> `CommError`, raised straight
out of `upsert_run`. No run is created. (The `if res else []` guard does not
help -- `{"ServerSettingsType": None}` is truthy.)

WHY IT LOOKED LIKE SUCCESS. `wandb/sync/sync.py` marks a run synced on

    if self._mark_synced and not self._view and finished:

i.e. on whether the LOCAL run finished, never on whether the upload worked. So
each failure still wrote `<run>.wandb.synced`, and the next `wandb sync` skipped
it as already done. Silent and self-sealing -- hence `--include-synced` in the
companion shell script.

THE FIX. Re-issue the same probe with correct null handling, so an absent type
yields `[]` rather than raising. `[]` is the value the code already uses for
"this server has no ServerSettings", and it must NOT be `[""]`: the list is only
truth-tested, and a truthy value makes `upsert_run` request a
`serverSettings { serverMessages }` fragment that this server cannot resolve --
which would fail again, differently.

The query is re-issued rather than short-circuited to `[]` so this stays correct
against a self-hosted server, where the type does exist.

Patched in this process only; nothing in site-packages is modified, so a wandb
upgrade neither clobbers the fix nor needs it reapplied -- check whether the
upstream bug is gone and then just use `wandb` directly again.

Usage -- a drop-in for the `wandb` executable:

    python wandb_fixed.py sync --include-synced <offline-run-dir>
    python wandb_fixed.py status
"""

from __future__ import annotations

import sys

from wandb.sdk.internal import internal_api

_PROBE = """
   query ProbeServerSettings {
       ServerSettingsType: __type(name: "ServerSettings") {
           fields { name }
        }
    }
"""


def _server_settings_introspection(self) -> None:
    """`internal_api.Api.server_settings_introspection`, null-safe."""
    if self._server_settings_type is not None:
        return
    res = self.gql(internal_api.gql(_PROBE))
    settings_type = res.get("ServerSettingsType") if res else None
    self._server_settings_type = (
        [field.get("name", "") for field in (settings_type.get("fields") or [{}])]
        if settings_type
        else []
    )


internal_api.Api.server_settings_introspection = _server_settings_introspection

# Imported after the patch so nothing captures the unpatched method first.
from wandb.cli.cli import cli  # noqa: E402  pylint: disable=wrong-import-position

if __name__ == "__main__":
    sys.exit(cli())
