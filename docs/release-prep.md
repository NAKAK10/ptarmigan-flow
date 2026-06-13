# Release Prep

This repository has two distribution paths:

- GitHub Pages site deployment from `.github/workflows/pages.yml`
- Signed and notarized macOS app zip from `.github/workflows/release-macos-app.yml`

Do not paste secret values into issues, pull requests, commits, or logs.

## GitHub Pages

The repository Pages setting must use GitHub Actions as its build and deploy source.
The workflow builds `site/`, transcodes `assets/usage-sample.mov` to
`_site/assets/usage-sample.mp4`, and deploys the artifact with
`actions/deploy-pages`.

The Pages workflow cannot deploy until the workflow file is present on the
branch that is allowed to run the deployment. If `main` is intentionally
off-limits, keep the changes on `dev` and wait to deploy until the project owner
chooses the branch or PR flow for publishing.

## Apple Developer Material

Prepare these before running the macOS release workflow:

- Apple Developer Program membership
- Developer ID Application certificate exported as a `.p12`
- App-specific password for the Apple ID used with notarization
- Apple Team ID

The workflow validates required GitHub Secrets before it builds the app. Missing
values fail early with `Missing GitHub secret: <name>`.

## Required GitHub Secrets

Set these repository secrets:

- `APPLE_CERTIFICATE_BASE64`: base64-encoded `.p12` certificate
- `APPLE_CERTIFICATE_PASSWORD`: password for the `.p12`
- `APPLE_TEAM_ID`: Apple Developer Team ID
- `APPLE_ID`: Apple ID email used for notarization
- `APPLE_APP_SPECIFIC_PASSWORD`: app-specific password for notarization

Set the base64 certificate secret without committing the certificate:

```bash
base64 -i DeveloperIDApplication.p12 \
  | gh secret set APPLE_CERTIFICATE_BASE64 --repo NAKAK10/ptarmigan-flow
```

Then set the remaining secrets. `gh` will prompt for each value:

```bash
gh secret set APPLE_CERTIFICATE_PASSWORD --repo NAKAK10/ptarmigan-flow
gh secret set APPLE_TEAM_ID --repo NAKAK10/ptarmigan-flow
gh secret set APPLE_ID --repo NAKAK10/ptarmigan-flow
gh secret set APPLE_APP_SPECIFIC_PASSWORD --repo NAKAK10/ptarmigan-flow
```

## Release Workflow

Create or choose an existing tag, then run the workflow manually:

```bash
gh workflow run release-macos-app.yml \
  --repo NAKAK10/ptarmigan-flow \
  --ref main \
  -f tag=v0.0.0
```

The workflow checks out the tag and builds `PtarmiganFlow.app` twice with
PyInstaller:

- `PtarmiganFlow-macos-arm64.zip` on the Apple Silicon `macos-15` runner
- `PtarmiganFlow-macos-x86_64.zip` on the Intel `macos-15-intel` runner

Each app is codesigned with the Developer ID certificate, notarized with Apple,
stapled, zipped, uploaded as a workflow artifact, and then attached to one draft
GitHub Release.

Before signing, the workflow validates that `moonshine_voice/libmoonshine.dylib`
and `libonnxruntime` contain the target architecture. The arm64 job repackages
the pinned `moonshine-voice==0.0.49` PyPI wheel. The Intel job checks out
`moonshine-ai/moonshine` at the matching tag with Git LFS, downloads the
official `onnxruntime-osx-x86_64` binary, source-builds Moonshine on
`macos-15-intel`, and packages the resulting dylibs into the release wheel
before the same native-library validation runs.

The downloadable app is a compact Moonshine build. It uses the release-only
dependency list in `packaging/macos/requirements-release.txt`, defaults new
configs to `stt.model = "moonshine:tiny"`, and does not ship the optional
Torch, Transformers, MLX, MLX-audio, VoxMLX, Granite, or Voxtral runtime
stacks. Those larger backends remain available through the normal CLI/Homebrew
environment.

Publish the draft release only after downloading the zips and checking the app on
clean Apple Silicon and Intel Macs. Publishing the release keeps the existing
Homebrew formula update workflow responsible for formula changes.
