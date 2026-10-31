# Autsunset NAS Deployment

The development source is this fork. Do not develop by copying replacement
modules into the running NAS container.

## Image construction

`deploy/Dockerfile.autsun` layers the fork over the official 1.3.19 runtime
published from upstream commit
`c6429547af711555e3d4d1b8d04b413572cb616c`. Its default base is pinned by
manifest digest. It rejects a different `pyproject.toml` rather than silently
using stale dependencies. If dependencies change, rebuild the full runtime
with the root Dockerfile or deliberately update the verified base.

Build from committed source only, not from a live workspace containing private
configuration:

```bash
revision=$(git rev-parse HEAD)
git archive --format=tar HEAD |
  docker build -f deploy/Dockerfile.autsun \
    --build-arg SOURCE_REVISION="$revision" \
    -t "maibot-autsun:1.3.19-${revision:0:8}" -
```

For an offline NAS, transfer and load the verified official runtime archive.
Verify its image identity and upstream revision before assigning a local base
tag. The same build can then use
`--build-arg MAIBOT_BASE=maibot-autsun:base-c6429547`. Keep the original
manifest digest in the private deployment receipt.

## Upgrade procedure

1. Record the running image, source hashes, Compose configuration, model
   assignments, mounts and networks. Keep the old image available locally.
2. Make private configuration copies and a consistent SQLite online backup.
   Never copy a live database without accounting for its WAL.
3. Test configuration and database initialization on isolated copies using
   `MAIBOT_INSTANCE_DIR` and `MAIBOT_INITIALIZE_ONLY=1`, with networking
   disabled. Explicitly call `config_manager.initialize()` and
   `initialize_database()`. Compare preserved configuration values and table
   row counts; inspect any changes before continuing.
4. Stop only the core service, then take fresh recovery copies before allowing
   the new application to migrate production data.
5. Select the versioned fork image in Compose. Remove only obsolete
   full-module patch mounts whose changes are already baked into the image.
   Preserve the old patch files, configuration, plugins, data, networks,
   certificates and model assignments.
6. Start only core with `docker compose up -d --no-deps --pull never core`.
   Do not recreate NapCat, embedding services or model workers.
7. Verify image revision, schema version, WebUI, plugin and transport status,
   embeddings, text and vision requests, and model image-budget behavior.

Rollback after a database or configuration migration requires stopping core
and restoring the matching recovery copies before starting the saved old
image. Merely switching the image back is not a complete rollback.

Deployment receipts and backups belong in private operational storage, never
in Git. GitHub pushes and upstream merges do not deploy the NAS automatically.
