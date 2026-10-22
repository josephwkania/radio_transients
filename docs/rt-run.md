# rt-run reference

The README has the quick start. This page covers the details.

## Installing

    pip install radio-transients      # or pipx install radio-transients
    rt-run install gpu                # or full, cpu, arm, full-cuda12.6, gpu-cuda12.6

`pip install --user` works on most clusters. On a distro Python marked
"externally managed" (Ubuntu 23.04+, Debian 12) use pipx or a venv instead.
The host needs Apptainer or Singularity, and Python 3.8 or newer.

With no variant, `rt-run install` picks arm on aarch64, full when `nvidia-smi`
is found, and cpu otherwise.

To install an image you already have:

    singularity run --app install radio_transients_gpu.sif    # from inside the image
    rt-run install --from radio_transients_gpu.sif            # from the host

Images built before `rt-run` existed have neither the `install` app nor the
label saying which variant they are, so name it:
`rt-run install gpu --from old.sif`.

## Commands

- `rt-run update` pulls the image again; the tags are rebuilt every week. It
  also takes the copy of `rt-run` inside the new image when that is at least as
  new as yours, so the launcher keeps pace with the images by itself.
- `rt-run list` shows the tools it linked: the programs the recipes build, and
  the scripts of the Python packages they install from source. Anything else in
  the image runs as `rt-run COMMAND`, e.g. `rt-run python script.py`.
- `rt-run uninstall` removes the links, and the image if `rt-run` pulled it.

## Digests and history

`rt-run list` also shows when the image was pulled, where from, when it was
built, and its digest: the registry's fingerprint for exactly that build.
The tags move every week; a digest never does, and

    rt-run install gpu --digest sha256:...

installs that same build again, here or anywhere. `rt-run` pulls by digest
whenever the registry will say what a tag points at, and `update` uses it to
download nothing when the tag hasn't moved.
`~/.local/share/radio_transients/history` keeps a line for every install and
update -- time, tag, build date, source, digest, signature check -- so you can
tell which image a result was produced with.

## Signatures

CI signs every image it publishes, with [cosign](https://docs.sigstore.dev/)'s
keyless signing: the signature says the image was built by
`build-test-push.yml` on `main`, with GitHub vouching for that, so there is
no key to leak. When `cosign` is installed, `rt-run install` and `update`
check that signature before pulling, in about a second, and refuse an image
without it; the pull then checks the download against the signed digest as it
arrives. Without `cosign` they warn and go ahead.

- `RT_VERIFY=require` refuses any image that can't be checked, including one
  installed `--from` a file.
- `RT_VERIFY=0` skips the check -- needed for images published before
  signing began (v0.3.0).

`rt-run list` says how the installed image fared. To check by hand:

    cosign verify \
      --certificate-identity https://github.com/josephwkania/radio_transients/.github/workflows/build-test-push.yml@refs/heads/main \
      --certificate-oidc-issuer https://token.actions.githubusercontent.com \
      ghcr.io/josephwkania/radio_transients:gpu

## How the links run

Each link runs `apptainer exec` (or `singularity exec`) on the installed image,
with `--nv` for the GPU images when the host has `nvidia-smi`. An argument
naming a path outside `$HOME`, `/tmp` and the current directory -- the value in
`--opt=/path` included -- has the filesystem it is on bind-mounted at the same
place, so paths mean the same inside and out: `/data` for `/data/obs/x.fil`,
or `/lustre/project` if that is where a network filesystem is mounted. Tools
can then reach the files around the ones they are given, as they would on the
host. Paths under the container's own system directories (`/usr`, `/etc`,
`/opt`, ...) are never bound over them.

Settings:

- `RT_BIND=/data,/scratch` binds paths it can't see, such as ones inside a
  config file.
- `RT_AUTOBIND=0` turns off automatic mounts, leaving only what `RT_BIND`
  names.
- `RT_HOME` is where the image, links and state live
  (`~/.local/share/radio_transients` by default).
- `RT_DEBUG=1` prints the command instead of running it.

`rt-run --help` and the top of `src/rt_run.py` list the rest.

Appending the bin directory to `PATH`, as the README does, keeps any host
command of the same name ahead of the container's.

## Safeguards

- `rt-run` refuses to run, or install, if anyone else could change its copy,
  its state, its links or the image -- they could then run code as you. That
  matters when `RT_HOME` is on group-writable project storage; the message
  says what to `chmod`. A group that is yours alone, as Ubuntu and Red Hat
  give each user, doesn't count.
- It won't install into a directory that already holds other files, and
  `uninstall` only touches a directory `rt-run` has marked as its own, so a
  mistyped `RT_HOME` can't cost you files.
- Installs, updates and uninstalls in one `RT_HOME` take turns.
- A pull that can't fit is refused before it starts.
- With `cosign` installed, an image CI didn't sign is refused before it is
  pulled.
- The image's own top-level directories, such as `/fetch`, are never mounted
  over.
- `install` names any tool that shares a name with a command already on your
  `PATH`, and says which of the two will run; `rt-run list` says the same for
  the `PATH` you have now. It never replaces a file it didn't make.
- On a host with no `python3`, the links say so rather than failing obscurely.

## Under Slurm

The links work in batch jobs as they do on a login node. On a two-GPU node
with Slurm's cgroup device limits, each job saw only the GPU it was given --
as GPU 0, so `-gpu_id 0` is right -- two jobs ran side by side on different
GPUs, and `scancel` stopped the container and everything in it.

What depends on the site:

- `--nv` is decided on the node a tool runs on, so CPU nodes don't get it.
- If apptainer is a module, a job that never runs `module load apptainer`
  still works: `rt-run` falls back to the apptainer that `install` or
  `update` last ran with.
- If jobs don't inherit your environment (`--export=NONE`), add the
  `export PATH=...` line to the script, or call the links by full path.
- A per-job `TMPDIR` outside `/tmp` is bind-mounted automatically.
- The image is 1 to 7 G. If your home quota can't take it, set `RT_HOME` to
  storage every node sees and that isn't purged, such as a project
  directory, for `rt-run install`. The links find it from then on without
  `RT_HOME` being set.
