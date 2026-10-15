#!/usr/bin/env python3
"""Run the tools in a radio_transients container as if they were installed on
the host.

    rt-run install [VARIANT] [--bin-dir DIR] [--from FILE.sif]
    rt-run update
    rt-run list
    rt-run uninstall
    rt-run TOOL [ARGS...]

`install` pulls an image and puts a link for each of its tools in a bin
directory. Every link points at a copy of this file, which runs the tool with
`apptainer exec` (or `singularity exec`). Any argument naming a host path --
the value in --opt=/path included -- has the filesystem it is on bind-mounted
at the same place in the container, unless Singularity mounts it already
($HOME, /tmp, $PWD).

The images carry this file too: `singularity run --app install IMAGE` installs
from an image you already have, and `install` and `update` take the image's
copy whenever it is at least as new as the one running, so the launcher keeps
pace with the images without a separate upgrade.

Environment:
    RT_HOME   where the image, links and tool list live
              (default: ~/.local/share/radio_transients)
    RT_SIF    run this image instead of the installed one
    RT_NV     1/0 forces --nv on/off (default: on for the GPU images when
              nvidia-smi is found)
    RT_BIND   extra comma-separated bind specs, e.g. /data,/scratch:/mnt
    RT_DEBUG  print the command instead of running it
    RT_REGISTRY_API  the registry's API (default: https://ghcr.io); the
              tests point it at a fake one
    RT_MOUNTINFO  the mount table to read (default: /proc/self/mountinfo);
              likewise for the tests

Standard library only. The host side runs on whatever python3 a login node
has, and scan() also runs on the python inside the images.
"""

import argparse
import contextlib
import fnmatch
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

__version__ = "0.1.0"

REGISTRY = "oras://ghcr.io/josephwkania/radio_transients"
REPOSITORY = "josephwkania/radio_transients"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

# What users type, mapped to the GHCR tag. The tags themselves work too.
VARIANTS = {
    "full": "latest",
    "full-cuda12.6": "latest-cuda12.6",
    "gpu": "gpu",
    "gpu-cuda12.6": "gpu-cuda12.6",
    "cpu": "cpu",
    "arm": "arm",
}
VARIANTS.update({tag: tag for tag in list(VARIANTS.values())})

# Images with nothing to give a GPU.
CPU_ONLY = ("cpu", "arm")

USAGE = """\
Usage:
  rt-run install [VARIANT] [--bin-dir DIR] [--from FILE.sif | --digest SHA]
  rt-run update            pull the image again; the tags are rebuilt weekly
  rt-run list              the tools the installed image puts on your PATH
  rt-run uninstall         remove the links, and the image if rt-run pulled it
  rt-run TOOL [ARGS...]    run any command in the image
  rt-run --version

VARIANT is one of
  full           everything, CUDA 11.8 (6.7 G)
  full-cuda12.6  everything, CUDA 12.6 (7.0 G)
  gpu            GPU tools only, CUDA 11.8 (6.5 G)
  gpu-cuda12.6   GPU tools only, CUDA 12.6 (6.8 G)
  cpu            CPU tools only (1.1 G)
  arm            CPU tools on aarch64 (1.0 G)
With no VARIANT, `install --from FILE` uses the one FILE was built as, and a
pull picks arm on aarch64, full when nvidia-smi is found, and cpu otherwise.

--from FILE.sif installs an image you already have instead of pulling one;
inside an image, `singularity run --app install IMAGE` does the same.

--digest sha256:... pulls that exact build of VARIANT, as `rt-run list` and
the history file record it, rather than whatever its tag points at now.
"""


class Error(Exception):
    """A failure to report in one line, without a traceback."""


class UsageError(Error):
    """A command line rt-run can't make sense of; exits with status 2, and
    shows the usage unless the message already says what to type."""

    def __init__(self, message, show_usage=True):
        super().__init__(message)
        self.show_usage = show_usage


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------


class Home:
    """The files rt-run keeps under RT_HOME.

    image     path of the installed image: RT_HOME/image.sif, or an image
              installed --from that could not be hard-linked there
    tag       the GHCR tag it was built as, which `update` pulls
    bin-dir   where the links are
    built     its build date, so `update` can tell whether anything changed
    commands  everything on the image's PATH, one per line
    tools     the commands that got a link
    rt-run    the copy of this file every link points at
    engine    the apptainer or singularity install and update ran with
    arrived   when the image was pulled, or installed --from a file
    origin    where from: the registry and tag, or the file
    digest    the registry's sha256 for the image pulled, which pins it
    history   a line per install and update, never rewritten: which image
              was in use when, for anyone asking what their results ran on
    """

    def __init__(self, root):
        self.root = root
        self.script = os.path.join(root, "rt-run")
        self.pulled = os.path.join(root, "image.sif")
        self.pending = self.pulled + ".new"

    @classmethod
    def from_env(cls):
        if os.environ.get("RT_HOME"):
            return cls(os.path.abspath(os.environ["RT_HOME"]))
        # Run through a link, this file is RT_HOME/rt-run, so it knows its
        # home even where RT_HOME isn't set -- a job's environment, say.
        me = os.path.realpath(__file__)
        here = os.path.dirname(me)
        if os.path.basename(me) == "rt-run" and os.path.isfile(
            os.path.join(here, "tag")
        ):
            return cls(here)
        data = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
        return cls(os.path.join(data, "radio_transients"))

    def path(self, name):
        return os.path.join(self.root, name)

    def get(self, name):
        try:
            with open(self.path(name), encoding="utf-8") as handle:
                return handle.read().strip()
        except FileNotFoundError:
            return ""

    def lines(self, name):
        return self.get(name).splitlines()

    def put(self, name, text):
        write_atomically(self.path(name), text)


def write_atomically(path, text, mode=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def remove(path):
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def own_image():
    """The image this process runs in, or None on the host."""
    for var in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def find_engine(home=None):
    """The full path of apptainer or singularity: from PATH, or else the one
    install or update last ran with. On a cluster the runtime is often a
    module, on PATH only once loaded -- which a job script may not do."""
    for name in ("apptainer", "singularity"):
        path = shutil.which(name)
        if path:
            return path
    recorded = home.get("engine") if home else ""
    if recorded and os.access(recorded, os.X_OK):
        return recorded
    raise Error(
        "neither apptainer nor singularity is on PATH; on a cluster, "
        "`module load apptainer` (or singularity) first"
    )


def capture(cmd):
    """Run cmd and return its stdout; its stderr goes straight to the user."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, check=False
    )
    if result.returncode != 0:
        raise Error("failed: " + " ".join(shlex.quote(c) for c in cmd[:4]) + " ...")
    return result.stdout


# ---------------------------------------------------------------------------
# What an image offers
# ---------------------------------------------------------------------------


def scan(labels_file="/.singularity.d/labels.json", dists=None):
    """Describe the image this runs in.

    Runs inside the image: the host sends the image's python the source of
    this and of the other functions in SCAN, so they may use only the
    standard library, and only json and os at module level.

    commands     [dir, name, kind] for every executable on PATH, in PATH
                 order; kind is "x" for a program (an ELF binary or a #!
                 script), "-" for anything else with its x bit set
    from_source  the scripts of every package pip installed from a local
                 checkout -- the ones the recipe cloned, not their
                 dependencies
    tag, built   the image's RT_TAG and build-date labels
    self         [path, version] of the rt_run module in the image, if any
    """
    labels = read_labels(labels_file)
    return {
        "commands": executables(),
        "from_source": installed_from_source(dists),
        "tag": labels.get("RT_TAG", ""),
        "built": labels.get("org.label-schema.build-date", ""),
        "self": this_module(),
    }


def executables():
    found, seen = [], set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory or directory in seen or not os.path.isdir(directory):
            continue
        seen.add(directory)
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                found.append([directory, name, kind_of(path)])
    return found


def kind_of(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(4)
    except OSError:
        return "-"
    return "x" if head == b"\x7fELF" or head[:2] == b"#!" else "-"


def installed_from_source(dists=None):
    if dists is None:
        try:
            import importlib.metadata  # pylint: disable=import-outside-toplevel

            dists = importlib.metadata.distributions()
        except ImportError:
            dists = []
    names = set()
    for dist in dists:
        # conda writes direct_url.json too; INSTALLER tells them apart.
        installer = (dist.read_text("INSTALLER") or "").strip()
        if installer == "pip" and dist.read_text("direct_url.json"):
            names.update(scripts_of(dist))
    return sorted(names)


def scripts_of(dist):
    for entry in dist.entry_points:
        if entry.group == "console_scripts":
            yield entry.name
    for file in dist.files or []:
        # RECORD paths are relative to site-packages: ../../../bin/NAME
        parts = str(file).split("/")
        if len(parts) == 5 and parts[:4] == ["..", "..", "..", "bin"]:
            if not parts[4].endswith(".pyc"):
                yield parts[4]


def read_labels(labels_file):
    try:
        with open(labels_file, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def this_module():
    try:
        # pylint: disable-next=import-outside-toplevel,import-self
        import rt_run

        return [rt_run.__file__, rt_run.__version__]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


SCAN = (
    scan,
    executables,
    kind_of,
    installed_from_source,
    scripts_of,
    read_labels,
    this_module,
)


def scan_image(engine, sif):
    code = "import json\nimport os\n\n"
    code += "\n".join(inspect.getsource(f) for f in SCAN)
    code += "\nprint(json.dumps(scan()))\n"
    # -I: nothing from the host's cwd or PYTHON* variables leaks into the scan.
    out = capture([engine, "exec", "--cleanenv", sif, "python", "-I", "-c", code])
    try:
        return json.loads(out)
    except ValueError as err:
        raise Error(f"could not list the commands in {sif}") from err


SYSTEM_DIRS = (
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/local/sbin",
    "/usr/games",
    "/usr/local/games",
)

# Generic names that would shadow the host's own, and rt-run itself.
GENERIC = (
    "python*",
    "pip",
    "pip3*",
    "conda*",
    "activate",
    "deactivate",
    "wheel",
    "idle*",
    "pydoc*",
    "2to3*",
    "*-config",
    "rt-run",
)


def should_link(directory, name, kind, from_source):
    """Whether a command gets a link: the programs the recipes build, and the
    scripts of the python packages they install from source.

    Not the system directories, not CUDA's or conda's own machinery, and not
    the hundreds of scripts the RT env gets with its dependencies (pylint,
    sphinx-build, ...), which would bury the real tools in `rt-run list`.
    `rt-run COMMAND` still reaches all of those.
    """
    if directory in SYSTEM_DIRS or directory.startswith(
        ("/usr/local/cuda", "/usr/local/nvidia", "/scif", "/.singularity.d")
    ):
        return False
    if directory.endswith(("/miniconda/bin", "/miniconda/condabin")):
        return False
    if directory.endswith("/envs/RT/bin"):
        # jupyter comes from conda, but the README lists it as a tool.
        wanted = from_source or name == "jupyter"
    else:
        wanted = kind == "x"
    return wanted and not any(fnmatch.fnmatchcase(name, p) for p in GENERIC)


def sort_out(found):
    """Split a scan into every command on the image's PATH, and those to link."""
    from_source = set(found["from_source"])
    commands, tools, seen = [], [], set()
    for directory, name, kind in found["commands"]:
        # The first directory on PATH wins, as it does when the tool runs.
        if name in seen:
            continue
        seen.add(name)
        commands.append(name)
        if should_link(directory, name, kind, name in from_source):
            tools.append(name)
    return commands, sorted(tools)


# ---------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------


def is_ours(link, target):
    return os.path.islink(link) and os.readlink(link) == target


def unlink_all(bin_dir, target):
    """Remove every link of ours from bin_dir, and nothing else."""
    with contextlib.suppress(FileNotFoundError):
        for name in os.listdir(bin_dir):
            path = os.path.join(bin_dir, name)
            if is_ours(path, target):
                os.remove(path)


def link_tools(bin_dir, tools, target):
    """Point a link per tool at target. Drop links of ours for tools the
    image no longer has; never touch a file rt-run did not make."""
    os.makedirs(bin_dir, exist_ok=True)
    wanted = set(tools)
    for name in os.listdir(bin_dir):
        path = os.path.join(bin_dir, name)
        if name != "rt-run" and name not in wanted and is_ours(path, target):
            os.remove(path)
    linked, skipped = [], []
    for name in tools:
        path = os.path.join(bin_dir, name)
        if os.path.lexists(path) and not is_ours(path, target):
            skipped.append(path)
            continue
        if not os.path.lexists(path):
            os.symlink(target, path)
        linked.append(name)
    # An rt-run already there -- pip's, say -- does the same job.
    path = os.path.join(bin_dir, "rt-run")
    if not os.path.lexists(path):
        os.symlink(target, path)
    return linked, skipped


def path_overlaps(tools, bin_dir, path):
    """The tools that share a name with a command elsewhere on PATH, as
    [(tool, that command, whether the link comes first)]. With bin_dir not on
    PATH yet, the other command comes first, as it will once bin_dir is
    appended."""

    def norm(directory):
        return os.path.realpath(os.path.expanduser(directory))

    dirs = [d for d in path.split(os.pathsep) if d]
    ours = norm(bin_dir)
    ours_at = next((i for i, d in enumerate(dirs) if norm(d) == ours), len(dirs))
    overlaps = []
    for tool in tools:
        for i, directory in enumerate(dirs):
            other = os.path.join(directory, tool)
            if norm(directory) == ours or not os.access(other, os.X_OK):
                continue
            if os.path.isfile(other):
                overlaps.append((tool, other, ours_at < i))
                break
    return overlaps


def report_overlaps(overlaps, bin_dir, limit=None):
    """Say which of the host's commands and the image's tools win."""
    theirs = [(t, o) for t, o, ours_first in overlaps if not ours_first]
    ours = [(t, o) for t, o, ours_first in overlaps if ours_first]
    if theirs:
        print()
        verb = "shares" if len(theirs) == 1 else "share"
        print(f"{len(theirs)} of the tools {verb} a name with a command of yours,")
        print("which runs first; `rt-run TOOL` runs the image's instead:")
        print_overlaps(theirs, limit)
    if ours:
        print()
        verb = "takes" if len(ours) == 1 else "take"
        print(f"{len(ours)} of the tools {verb} over from a command of yours, as")
        print(f"{bin_dir} comes first on your PATH. Run yours by its full")
        print("path, or move that directory to the end of PATH:")
        print_overlaps(ours, limit)


def print_overlaps(overlaps, limit):
    shown = overlaps if limit is None else overlaps[:limit]
    for tool, other in shown:
        print(f"    {tool:24} {other}")
    if len(shown) < len(overlaps):
        print(f"    ... and {len(overlaps) - len(shown)} more; `rt-run list` shows all")


def place_image(home, source):
    """Install an existing image without copying gigabytes: hard-link it into
    RT_HOME, or where that can't be done, use it where it is."""
    remove(home.pending)
    try:
        os.link(source, home.pending)
        return home.pending
    except OSError:
        return os.path.abspath(source)


def resolve_digest(tag):
    """The digest the tag points at now, or "" if the registry can't say.

    Two small requests, no login: an anonymous pull token, then the tag's
    manifest headers, which carry its digest."""
    # Imported here, not above: it brings in http and ssl, and every tool
    # run starts this module, while only install and update need them.
    import urllib.request  # pylint: disable=import-outside-toplevel

    api = os.environ.get("RT_REGISTRY_API", "https://ghcr.io")
    try:
        url = f"{api}/token?scope=repository:{REPOSITORY}:pull"
        with urllib.request.urlopen(url, timeout=30) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(
            f"{api}/v2/{REPOSITORY}/manifests/{tag}",
            method="HEAD",
            headers={"Authorization": f"Bearer {token}", "Accept": OCI_MANIFEST},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            digest = response.headers.get("Docker-Content-Digest", "")
    except (OSError, ValueError, KeyError):
        return ""
    return digest if DIGEST.fullmatch(digest) else ""


def digest_for(tag):
    digest = resolve_digest(tag)
    if not digest:
        print(
            "rt-run: the registry didn't say which image the tag is now; "
            "pulling by tag, with no digest to record",
            file=sys.stderr,
        )
    return digest


def pull(engine, tag, dest, digest=""):
    """Pull by digest when there is one -- exactly the image resolved, even
    if the weekly rebuild moves the tag meanwhile -- or else by tag."""
    remove(dest)
    ref = f"{REGISTRY}@{digest}" if digest else f"{REGISTRY}:{tag}"
    print(f"Pulling {ref}", flush=True)
    # The tags are rebuilt weekly, so a cached copy is almost never reused,
    # and caching would keep a second copy of a multi-gigabyte image.
    cmd = [engine, "pull", "--disable-cache", dest, ref]
    if subprocess.run(cmd, check=False).returncode != 0:
        raise Error(f"could not pull {ref}")
    return dest


def version_key(version):
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, ValueError):
        return ()


def own_source():
    with open(os.path.abspath(__file__), encoding="utf-8") as handle:
        return handle.read()


def newest_script(engine, image, found):
    """This file's text for RT_HOME/rt-run: the image's copy if it is at least
    as new as the one running, so launcher and image come from one build."""
    theirs = found.get("self")
    if engine and theirs and version_key(theirs[1]) >= version_key(__version__):
        return capture([engine, "exec", "--cleanenv", image, "cat", theirs[0]])
    return own_source()


def default_variant():
    if platform.machine() == "aarch64":
        return "arm"
    if shutil.which("nvidia-smi"):
        return "full"
    return "cpu"


def check_arch(tag):
    machine = platform.machine()
    if machine == "aarch64" and tag != "arm":
        raise Error("this is an aarch64 machine; only the arm variant runs here")
    if machine != "aarch64" and tag == "arm":
        raise Error(f"the arm variant is for aarch64 machines, and this is {machine}")


@contextlib.contextmanager
def cleanup_on_failure(home):
    """Don't leave gigabytes behind if a pull, or anything after it, fails."""
    try:
        yield
    except BaseException:
        remove(home.pending)
        raise


# The head of RT_HOME/rt-run, the copy every link points at. sh runs the
# lines down to the closing quotes: it checks for python3, then hands it this
# file under the name it was called by. To python they are only a string. A
# host without python3 then gets a sentence instead of env's "No such file".
LAUNCHER = """\
#!/bin/sh
''':'
command -v python3 >/dev/null 2>&1 || {
    echo "rt-run: needs python3 (3.8 or newer) on this machine" >&2
    exit 127
}
exec python3 "$0" "$@"
'''
"""


def as_launcher(source):
    """This module's source, headed by LAUNCHER instead of its #! line."""
    if source.startswith(LAUNCHER):
        return source
    if source.startswith("#!"):
        source = source.split("\n", 1)[1]
    return LAUNCHER + source


def install_script(home, engine, image, found):
    script = as_launcher(newest_script(engine, image, found))
    write_atomically(home.script, script, mode=0o755)


def activate(home, image, tag, bin_dir, found):
    """Make image the installed one and relink. install_script() first."""
    commands, tools = sort_out(found)
    if not tools:
        raise Error(f"found no tools in {image}")

    if image == home.pending:
        # Reinstalling --from the file already installed links it to itself.
        if os.path.exists(home.pulled) and os.path.samefile(image, home.pulled):
            remove(image)
        else:
            os.replace(image, home.pulled)
        image = home.pulled
    elif home.get("image") == home.pulled:
        remove(home.pulled)  # replaced by an image used in place

    old_bin = home.get("bin-dir")
    if old_bin and old_bin != bin_dir:
        unlink_all(old_bin, home.script)  # the links move, not multiply
    home.put("image", image)
    home.put("tag", tag)
    home.put("bin-dir", bin_dir)
    home.put("built", found.get("built", ""))
    home.put("commands", "".join(c + "\n" for c in commands))
    home.put("tools", "".join(t + "\n" for t in tools))
    linked, skipped = link_tools(bin_dir, tools, home.script)

    for path in skipped:
        print(f"rt-run: skipping {path}, which rt-run did not make", file=sys.stderr)
    built = found.get("built")
    print(f"Installed radio_transients:{tag}" + (f", built {built}" if built else ""))
    if image != home.pulled:
        print(f"Using {image} where it is; rt-run will not delete it.")
    print(f"{len(linked)} tools linked in {bin_dir}; `rt-run list` shows them.")
    if own_image():
        # This PATH is the container's, and its python3 says nothing of the
        # host's: say what to check there instead.
        print()
        print("On the host, the links need python3 (3.8 or newer), and")
        print("`rt-run list` there says whether any tool shares a name with a")
        print("command of yours.")
    else:
        report_overlaps(
            path_overlaps(linked, bin_dir, os.environ.get("PATH", "")),
            bin_dir,
            limit=10,
        )
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        print()
        print("Add them to your PATH, e.g. in ~/.bashrc:")
        print(f'    export PATH="$PATH:{bin_dir}"')
        print("Appending, as here, keeps any host command of the same name first.")


def image_source(home, inside, source):
    """The engine to run images with, and the existing image to install, if
    any. Inside an image there is no engine, and the image is that one."""
    if not inside:
        if source and not os.path.isfile(source):
            raise Error(f"no such image: {source}")
        return find_engine(home), source
    if source:
        raise UsageError(
            "inside an image, install uses that image; drop --from",
            show_usage=False,
        )
    if not os.path.isfile(inside):
        raise Error(
            f"{inside} isn't visible inside the container; mount its "
            f"directory: singularity run -B {os.path.dirname(inside)} "
            f"--app install {inside}"
        )
    return None, inside


def record(home, action, origin, digest=""):
    """Note when the image arrived and where from, and add it to history."""
    when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    home.put("arrived", when)
    home.put("origin", origin)
    home.put("digest", digest)
    fields = [when, action, f"radio_transients:{home.get('tag')}"]
    fields += [f"built {home.get('built') or 'unknown'}", origin, digest or "-"]
    with open(home.path("history"), "a", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")


def requested_tag(opts, source):
    """The tag asked for; the default, when pulling; or "" when the image
    being installed --from will say which it is."""
    if opts.digest:
        if source:
            raise UsageError(
                "--digest pulls an image; it can't go with --from, or run "
                "inside an image",
                show_usage=False,
            )
        if not DIGEST.fullmatch(opts.digest):
            raise UsageError(
                f"{opts.digest} isn't a digest; they are sha256: and 64 hex digits",
                show_usage=False,
            )
        if not opts.variant:
            raise UsageError(
                "name the variant the digest is a build of, "
                f"e.g. rt-run install gpu --digest {opts.digest}",
                show_usage=False,
            )
    if opts.variant:
        tag = VARIANTS.get(opts.variant, "")
        if not tag:
            raise UsageError(f"unknown variant '{opts.variant}'")
        return tag
    return "" if source else VARIANTS[default_variant()]


def cmd_install(argv):
    parser = argparse.ArgumentParser(
        prog="rt-run install",
        usage="rt-run install [VARIANT] [--bin-dir DIR] "
        "[--from FILE.sif | --digest SHA]",
        epilog="`rt-run --help` lists the variants.",
    )
    parser.add_argument("variant", nargs="?")
    parser.add_argument("--bin-dir")
    parser.add_argument("--from", dest="source")
    parser.add_argument("--digest")
    opts = parser.parse_args(argv)

    home = Home.from_env()
    inside = own_image()
    engine, source = image_source(home, inside, opts.source)

    tag = requested_tag(opts, source)
    if tag:
        check_arch(tag)

    os.makedirs(home.root, exist_ok=True)
    bin_dir = os.path.abspath(
        opts.bin_dir or home.get("bin-dir") or home.path("bin")
    )  # a reinstall keeps the bin dir an earlier --bin-dir chose
    digest = ""
    with cleanup_on_failure(home):
        if source:
            image = place_image(home, source)
        else:
            digest = opts.digest or digest_for(tag)
            image = pull(engine, tag, home.pending, digest)
        found = scan() if inside else scan_image(engine, image)
        if not tag:
            tag = found.get("tag", "")
            if not tag:
                raise UsageError(
                    f"can't tell which variant {source} is; name it, "
                    f"e.g. rt-run install gpu --from {source}",
                    show_usage=False,
                )
            check_arch(tag)
        install_script(home, engine, image, found)
        activate(home, image, tag, bin_dir, found)
    origin = os.path.abspath(source) if source else f"{REGISTRY}:{tag}"
    record(home, "install", origin, digest)
    if engine:
        home.put("engine", engine)
    return 0


def cmd_update():
    if own_image():
        raise Error("update pulls an image, so run it on the host: rt-run update")
    home = Home.from_env()
    tag = home.get("tag")
    if not tag:
        raise Error("nothing installed yet; run `rt-run install`")
    engine = find_engine(home)
    digest = digest_for(tag)
    if digest and digest == home.get("digest") and os.path.isfile(home.get("image")):
        print(f"radio_transients:{tag} is already up to date: {digest}")
        return 0
    before = home.get("built")
    with cleanup_on_failure(home):
        image = pull(engine, tag, home.pending, digest)
        found = scan_image(engine, image)
        install_script(home, engine, image, found)
        activate(home, image, tag, home.get("bin-dir"), found)
    record(home, "update", f"{REGISTRY}:{tag}", digest)
    home.put("engine", engine)
    if before and before == found.get("built"):
        print("(The image was already up to date.)")
    return 0


def cmd_list():
    home = Home.from_env()
    if not home.get("tag"):
        raise Error("nothing installed yet; run `rt-run install`")
    tools = home.lines("tools")
    print(f"radio_transients:{home.get('tag')}")
    print(f"  built      {home.get('built') or 'unknown'}")
    origin = home.get("origin")
    if origin:  # not recorded by installs older than this
        verb = "pulled" if origin.startswith("oras://") else "installed"
        print(f"  {verb:10} {home.get('arrived')}, from {origin}")
    digest = home.get("digest")
    if digest:
        print(f"  digest     {digest}")
    if origin:
        print(f"  history    {home.path('history')}")
    if digest:
        print()
        print("To install exactly this image again, anywhere:")
        print(f"    rt-run install {home.get('tag')} --digest {digest}")
    print()
    print(f"Linked in {home.get('bin-dir')}:")
    width = max((len(t) for t in tools), default=0) + 2
    columns = max(1, shutil.get_terminal_size().columns // width)
    rows = -(-len(tools) // columns)
    for row in range(rows):
        print("".join(t.ljust(width) for t in tools[row::rows]).rstrip())
    print()
    print("Anything else in the image runs as `rt-run COMMAND`, e.g. `rt-run python`.")
    bin_dir = home.get("bin-dir")
    report_overlaps(path_overlaps(tools, bin_dir, os.environ.get("PATH", "")), bin_dir)
    return 0


def cmd_uninstall():
    home = Home.from_env()
    if not home.get("tag"):
        raise Error(f"nothing installed at {home.root}")
    bin_dir, image = home.get("bin-dir"), home.get("image")
    unlink_all(bin_dir, home.script)
    for name in ("image", "tag", "bin-dir", "built", "commands", "tools", "engine"):
        remove(home.path(name))
    for name in ("arrived", "origin", "digest", "history"):
        remove(home.path(name))
    for path in (home.pulled, home.pending, home.script):
        remove(path)
    for directory in (bin_dir, home.root):
        with contextlib.suppress(OSError):
            os.rmdir(directory)
    print(f"Removed radio_transients from {home.root}")
    if image and image != home.pulled:
        print(f"Left {image} alone; rt-run did not pull it.")
    return 0


# ---------------------------------------------------------------------------
# Running a tool
# ---------------------------------------------------------------------------

# Never bound over the container's own: a host /usr/lib in place of the
# image's would break every tool in it. RT_BIND can still force one.
CONTAINER_OWNED = (
    "/bin",
    "/boot",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/opt",
    "/root",
    "/run",
    "/sbin",
    "/usr",
    "/var",
)


def under(path, roots):
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def mount_points(mountinfo):
    """Where filesystems are mounted, from the kernel's table."""
    points = set()
    try:
        with open(mountinfo, encoding="utf-8", errors="surrogateescape") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) > 4:
                    # Spaces and the like come octal-escaped: \040.
                    points.add(
                        re.sub(
                            r"\\([0-7]{3})",
                            lambda m: chr(int(m.group(1), 8)),
                            fields[4],
                        )
                    )
    except OSError:
        pass
    return points


def filesystem_root(path, points):
    """What a bind mount for path should cover: the deepest filesystem
    mounted at or above it, or, for a path on the root filesystem, the
    directory under / it is in -- /data for /data/obs1. Taking the deepest
    mount also catches network filesystems mounted inside another, which a
    mount of the outer one might not carry along."""
    parts = path.strip("/").split("/")
    for i in range(len(parts), 0, -1):
        candidate = "/" + "/".join(parts[:i])
        if candidate in points:
            return candidate
    return "/" + parts[0]


class Binds:
    """The -B mounts a tool's arguments need.

    A path argument mounts the whole filesystem it is on (filesystem_root),
    not just its own directory: tools reach beyond what they are given --
    ../calibration next to an observation, files a candidate list or config
    names, symlinked directories found walking a tree, globs a script
    expands -- and sites configure their bind paths the same way. The cost
    is that the container sees more of the host; it already runs as you,
    with your home mounted. Mounting a parent of your home or working
    directory, as /home can be, worked with SingularityCE 4.1.
    RT_BIND covers paths no argument names.
    """

    def __init__(self, mountinfo=None):
        # RT_MOUNTINFO stands in for the kernel's table in the tests.
        self.points = mount_points(
            mountinfo or os.environ.get("RT_MOUNTINFO") or "/proc/self/mountinfo"
        )
        # Paths the container sees without help.
        self.mounted = [os.path.expanduser("~"), "/tmp", "/var/tmp", "/proc", "/sys"]
        self.mounted += ["/dev"]
        # Singularity mounts the working directory, unless that is / itself.
        # $PWD spells it with symlinks kept, but only if it is current: a
        # program that changed directory may have passed on a stale one.
        cwd = [os.getcwd()]
        pwd = os.environ.get("PWD", "")
        with contextlib.suppress(OSError):
            if pwd and pwd != cwd[0] and os.path.samefile(pwd, cwd[0]):
                cwd.append(pwd)
        self.mounted += [d for d in cwd if d != "/"]
        self.paths = []

    def add(self, path):
        if path == "/" or path in self.paths:
            return
        if under(path, self.mounted) or under(path, CONTAINER_OWNED):
            return
        self.paths.append(path)

    def consider(self, arg):
        if not arg or arg.startswith("-"):
            return
        if os.path.isdir(arg):
            directory = arg
        elif os.path.exists(arg):
            directory = os.path.dirname(arg) or "."
        elif "/" in arg and os.path.isdir(os.path.dirname(arg) or "/"):
            directory = os.path.dirname(arg) or "/"  # an output yet to be written
        else:
            return
        # As spelled, normalised; and with symlinks resolved, as a symlink can
        # point somewhere unmounted, and its target must be there too. What
        # the container sees already stays as it is: widening it first would
        # turn a file in $HOME into a mount of all of /home.
        for path in (os.path.abspath(directory), os.path.realpath(directory)):
            if not under(path, self.mounted):
                self.add(filesystem_root(path, self.points))

    def args(self):
        out = []
        for path in self.paths:
            out += ["-B", path]
        for spec in os.environ.get("RT_BIND", "").split(","):
            if spec:
                out += ["-B", spec]
        return out


# What sh assumes when PATH is unset.
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def cmd_run(tool, args):
    if own_image():
        os.execvp(tool, [tool] + args)  # already in the image

    # A Slurm job started with --export=NONE has no PATH at all, and without
    # one the runtime can't find the host's nvidia-smi to mount for --nv, nor
    # apptainer some of its own helpers.
    if not os.environ.get("PATH"):
        os.environ["PATH"] = DEFAULT_PATH

    home = Home.from_env()
    engine = find_engine(home)
    sif, tag = os.environ.get("RT_SIF"), ""
    if sif:
        if not os.path.isfile(sif):
            raise Error(f"image not found: {sif}")
    else:
        sif, tag = home.get("image"), home.get("tag")
        if not sif:
            raise Error("no image installed; run `rt-run install` (or set RT_SIF)")
        if not os.path.isfile(sif):
            raise Error(f"the installed image {sif} is gone; run `rt-run update`")
        commands = home.lines("commands")
        # --nv brings the host's nvidia-smi and the like in at run time, so
        # the image's own list of commands never has them.
        driver_tool = tool.startswith("nvidia-")
        if "/" not in tool and commands and tool not in commands and not driver_tool:
            raise Error(
                f"'{tool}' is not in the installed {tag} image. `rt-run list` "
                "shows what is; `rt-run install full` has every tool."
            )

    binds = Binds()
    # A scheduler may point TMPDIR at a job's own local disk, which tools
    # then write to without it ever appearing among their arguments.
    binds.consider(os.environ.get("TMPDIR", ""))
    for arg in args:
        binds.consider(arg)
        if arg.startswith("-") and "=" in arg:
            binds.consider(arg.split("=", 1)[1])

    nv = []
    mode = os.environ.get("RT_NV", "")
    if mode == "1" or (
        mode != "0" and tag not in CPU_ONLY and shutil.which("nvidia-smi")
    ):
        nv = ["--nv"]

    cmd = [engine, "exec"] + nv + binds.args() + [sif, tool] + list(args)
    if os.environ.get("RT_DEBUG"):
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    os.execvp(engine, cmd)
    return 127  # not reached


# ---------------------------------------------------------------------------


def cmd_help():
    print(USAGE)
    return 0


def cmd_version():
    print(f"rt-run {__version__}")
    return 0


NO_ARGS = {
    "update": cmd_update,
    "list": cmd_list,
    "uninstall": cmd_uninstall,
    "help": cmd_help,
    "-h": cmd_help,
    "--help": cmd_help,
    "--version": cmd_version,
}


def dispatch(argv):
    # Called through a link named after a tool, or as `rt-run COMMAND`.
    name = os.path.basename(argv[0])
    if not name.startswith(("rt-run", "rt_run")):
        return cmd_run(name, argv[1:])
    if len(argv) < 2:
        raise UsageError("which command?")
    command, rest = argv[1], argv[2:]
    if command == "install":
        return cmd_install(rest)
    if command in NO_ARGS:
        if rest:
            raise UsageError(f"{command} takes no arguments")
        return NO_ARGS[command]()
    return cmd_run(command, rest)


def main(argv=None):
    try:
        return dispatch(sys.argv if argv is None else argv)
    except UsageError as err:
        print(f"rt-run: {err}", file=sys.stderr)
        if err.show_usage:
            print(f"\n{USAGE}", file=sys.stderr)
        return 2
    except Error as err:
        print(f"rt-run: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
