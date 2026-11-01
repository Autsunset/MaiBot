# Autsunset Fork Workflow

## Repository ownership

- Working copy: `/home/autsun/project/MaiBot`
- Personal remote: `origin`, repository `Autsunset/MaiBot`
- Official remote: `upstream`, repository `Mai-with-u/MaiBot`
- Working branch: `main`

The originally supplied `MaiM-with-u/MaiBot` address redirects to
`Mai-with-u/MaiBot`. The existing personal fork was reused rather than replaced.
Existing remote branches are preserved.

All future core development belongs in this working copy. Keep source changes
reviewable, test related behavior, commit, and push to the personal fork.
Plugins remain in their own repositories.

## Manual upstream updates

No scheduled merge or automatic overwrite is configured.

```bash
cd ~/project/MaiBot
git status --short
git fetch upstream
git log --oneline main..upstream/main
git diff --stat main...upstream/main
git merge upstream/main
# Resolve conflicts deliberately and run the relevant tests.
git push origin main
```

Start with a clean working tree; commit unfinished work on an appropriate branch
before merging. Do not force-push, discard personal changes, or push to upstream.
If a merge needs to be abandoned, inspect and obtain approval before discarding
any unresolved work.

For slow GitHub access, use the local proxy per command rather than changing
global Git configuration:

```bash
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 git fetch upstream
```

## Initial migration on 2026-10-31

- Running NAS image source revision:
  `ff0fe42500f5f7581e9655d1e78379407233207a` (1.3.11).
- Official main imported into this working copy:
  `c6429547af711555e3d4d1b8d04b413572cb616c` (1.3.19).
- The old fork main was 378 commits behind and had no exclusive commits, so the
  initial synchronization was a normal fast-forward.
- Only source files were compared; production configuration, credentials,
  databases, logs and chat images were not imported.

The actual NAS source differences from its image revision were three files:

1. `src/A_memorix/core/runtime/admin_contracts.py`: permit the existing
   `list_vector_spaces` and `delete_vector_space` administrative actions.
2. `src/A_memorix/host_service.py`: command-result values take precedence over
   generic startup status when merging a runtime-admin response.
3. `pytests/A_memorix_test/test_admin_contracts.py`: verify action dispatch.

These are small carried-forward compatibility patches. A_Memorix implementation
ownership remains governed by `src/A_memorix/MODIFICATION_POLICY.md`; reconcile
equivalent fixes with its upstream `MaiBot_branch` when practical.

The two matching NAS host patch files already had the same contents as the
running files, although their Compose bind mounts were not active. Migration
does not activate those mounts or recreate the production container.

`CLAUDE.md` is an unchanged symlink to `AGENTS.md`, not a separate custom source
modification. A generated built-in-plugin `config.toml` is runtime configuration
and is deliberately excluded.

Earlier image normalization, task hard timeouts and general image limiting are
already part of official source. Do not reapply old complete-file copies and
accidentally revert newer upstream changes.

The historical branch `origin/fix/vision-recognition-failure-state` is also
preserved. Its old upstream PR 2157 was closed without merging; it was not part
of the running stock image after the 2026-10-19 upgrade. Do not silently
cherry-pick that older eight-file implementation over the newer architecture.
It remains available in this same repository for a deliberate feature-by-feature
port and review.

## Request image-budget work

The image-budget draft was moved into this repository from the fnOS maintenance
workspace. It reuses one shared image limiter while retaining the original
Maisaka import path and ordinary-image exemptions.

- Discovery API: eight total images, including candidate collages.
- Verified other model identifiers: a conservative 32-image request budget.
- Local MiniCPM-V-4.7-1B: single-image operation.
- Other model identifiers: unchanged, except for the shared Discovery route.
- Limits are applied afresh for each model attempt and checked again after
  long-screenshot segmentation. Original history objects are not mutated.
- Screenshot splitting respects the remaining budget and resizes the whole
  image when necessary, rather than retaining only its trailing segments.

These are observed deployment policies, not advertised maximum capacities.
Re-verify them when changing providers. Text already surrounding omitted images
is retained; the limiter does not invent descriptions or erase history files.

## Production separation

Updating this repository or pushing GitHub does not upgrade the NAS. The NAS
still runs its previous image and existing runtime configuration until a
separate deployment is approved and verified. Review database migrations and
the prepared Compose mounts before deploying a newer upstream version.

Never commit runtime TOML files, API tokens, session cookies, databases, request
snapshots, real chat images or model weights.

## GitHub Actions in the fork

The official Docker Hub publishing workflows (Main and Dev) and Crowdin
translation synchronization are restricted to `Mai-with-u/MaiBot` at job level.
They are skipped in `Autsunset/MaiBot`; ordinary validation workflows keep their
existing triggers and behavior.

The inherited publishing jobs previously failed before building images because
the fork had no Docker Hub publishing credentials. Crowdin also had no project
ID or token, and its scheduled `r-dev` checkout targeted a branch absent from
the fork. These are publishing and synchronization configuration failures, not
MaiBot test results or evidence of a failed NAS deployment.

Keep the repository guards when merging official updates. Fork-specific image
publishing, if needed later, requires a deliberate workflow change and separate
credentials. Do not disable code checks or account-wide failure notifications to
hide unrelated CI failures. Historical failed runs remain in GitHub's history;
these guards affect new runs only.
