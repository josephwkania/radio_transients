"""Stand-ins for test_rt_run.py: a stub container runtime, a fake registry,
and a World -- a HOME, PATH and RT_HOME under a scratch directory -- to run
rt-run in, as a subprocess, the way a shell would.
"""

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
RT_RUN = SRC / "rt_run.py"
sys.path.insert(0, str(SRC))

import rt_run  # noqa: E402  pylint: disable=wrong-import-position

E = "/usr/local/miniconda/envs/RT/bin"
REGISTRY = "oras://ghcr.io/josephwkania/radio_transients"

# An image's PATH, as scan() reports it: [dir, name, kind] in PATH order.
LISTING = [
    [E, "python", "x"],
    [E, "python3.10", "x"],
    [E, "python3-config", "x"],
    [E, "pip", "x"],
    [E, "rt-run", "x"],
    [E, "your_header.py", "x"],
    [E, "jupyter", "x"],
    [E, "pylint", "x"],
    [E, "xz", "x"],
    ["/usr/local/miniconda/condabin", "conda", "x"],
    ["/usr/local/bin", "heimdall", "x"],
    ["/usr/local/bin", "your_header.py", "x"],
    ["/usr/local/bin", "psrdada.pc", "-"],
    ["/usr/local/cuda/bin", "nvcc", "x"],
    ["/usr/bin", "ls", "x"],
    ["/bin", "sh", "x"],
]
STUB = r"""#!{python}
import json, os, subprocess, sys

state = os.environ["STUB_DIR"]
args = sys.argv[1:]


def read(name, default=""):
    try:
        with open(os.path.join(state, name)) as handle:
            return handle.read()
    except FileNotFoundError:
        return default


if args[0] == "pull":
    dest, uri = [a for a in args[1:] if not a.startswith("-")]
    with open(os.path.join(state, "pulls"), "a") as handle:
        handle.write(uri + "\n")
    if read("pull_fails"):
        sys.exit(1)
    with open(dest, "w") as handle:
        json.dump({{"uri": uri, "built": read("built").strip()}}, handle)
elif args[0] == "exec" and args[1] == "--cleanenv":
    sif, cmd = args[2], args[3:]
    if cmd[0] == "cat":
        print(read("image_rt_run"), end="")
    elif read("real_scan_path"):
        env = {{"PATH": read("real_scan_path")}}
        sys.exit(subprocess.run([sys.executable] + cmd[1:], env=env).returncode)
    else:
        found = json.loads(read("scan.json"))
        try:
            with open(sif) as handle:
                found["built"] = json.load(handle).get("built", "")
        except ValueError:
            pass
        print(json.dumps(found))
elif args[0] == "exec":
    print("\n".join(["exec"] + args[1:]))
else:
    sys.exit(99)
"""

# cosign, as rt-run runs it: `verify ... IMAGE@DIGEST`. Records each call,
# and finds every image signed but those the "unsigned" file lists.
COSIGN = r"""#!{python}
import os, sys

state = os.environ["STUB_DIR"]
with open(os.path.join(state, "cosign_calls"), "a") as handle:
    handle.write("\t".join(sys.argv[1:]) + "\n")
try:
    with open(os.path.join(state, "unsigned")) as handle:
        unsigned = handle.read().split()
except FileNotFoundError:
    unsigned = []
if sys.argv[-1].split("@")[-1] in unsigned:
    print("Error: no signatures found", file=sys.stderr)
    print("error during command execution: no signatures found", file=sys.stderr)
    sys.exit(1)
print('[{{"critical": {{}}}}]')
"""


def fake_digest(tag, build):
    return "sha256:" + hashlib.sha256(f"{tag} {build}".encode()).hexdigest()


class FakeRegistry(http.server.ThreadingHTTPServer):
    """ghcr.io's anonymous-token and manifest endpoints, which is all
    resolve_digest asks. Each tag points at a digest until move() rebuilds
    it, as the weekly CI run does."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), RegistryHandler)
        self.builds = 0
        self.size = 1000  # bytes in each image
        tags = set(rt_run.VARIANTS.values())
        self.digests = {tag: fake_digest(tag, 0) for tag in tags}
        # A short poll, as stop() waits out one per test.
        serve = threading.Thread(target=self.serve_forever, args=(0.02,), daemon=True)
        serve.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def move(self, tag="latest"):
        self.builds += 1
        self.digests[tag] = fake_digest(tag, self.builds)
        return self.digests[tag]

    def stop(self):
        self.shutdown()
        self.server_close()


class RegistryHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # pylint: disable=invalid-name
        prefix = f"/v2/{rt_run.REPOSITORY}/manifests/"
        if self.path == f"/token?scope=repository:{rt_run.REPOSITORY}:pull":
            self.answer(200, b'{"token": "t"}')
        elif not self.path.startswith(prefix):
            self.answer(404)
        elif self.headers.get("Authorization") != "Bearer t":
            self.answer(401)
        else:
            start = len(prefix)
            self.manifest(self.path[start:])

    def manifest(self, ref):
        """A tag's manifest, or a digest's: the digest in a header, the
        image's size in its one layer."""
        digests = self.server.digests
        digest = digests.get(ref) or (ref if ref in digests.values() else None)
        if self.headers.get("Accept") != rt_run.OCI_MANIFEST or not digest:
            self.answer(404)
            return
        body = json.dumps({"layers": [{"size": self.server.size}]}).encode()
        self.answer(200, body, {"Docker-Content-Digest": digest})

    def answer(self, code, body=b"", headers=None):
        self.send_response(code)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, *args):  # pylint: disable=arguments-differ
        pass


class Result:
    def __init__(self, proc):
        self.rc = proc.returncode
        self.out = proc.stdout

    def lines(self):
        return self.out.splitlines()

    def binds(self):
        lines = self.lines()
        return [lines[i + 1] for i, line in enumerate(lines) if line == "-B"]

    def tail(self, n):
        return self.lines()[-n:]


class World:
    """A HOME, a stub runtime and the paths rt-run will use, under root."""

    def __init__(self, root, registry):
        self.root = root
        self.registry = registry
        self.home = root / "home"
        self.rt = self.home / ".local/share/radio_transients"
        self.bin = self.rt / "bin"
        self.state = root / "state"
        stub, gpu = root / "stub", root / "gpu"
        for directory in (self.home, self.state, stub, gpu):
            directory.mkdir()
        (stub / "apptainer").write_text(STUB.format(python=sys.executable))
        (gpu / "nvidia-smi").write_text("#!/bin/sh\necho 'GPU 0: stub'\n")
        for path in (stub / "apptainer", gpu / "nvidia-smi"):
            path.chmod(0o755)
        self.listing()
        self.mounts()
        self.built("Monday_7_September_2026")

    @property
    def path(self):
        """The stub runtime first; an nvidia-smi only if this machine has one.
        No cosign but the stub, and that only once cosign() puts it there."""
        host = os.environ["PATH"].split(os.pathsep)
        host = [d for d in host if not os.path.exists(os.path.join(d, "cosign"))]
        return os.pathsep.join([str(self.root / "stub")] + host)

    @property
    def gpu_path(self):
        """A PATH that always finds an nvidia-smi."""
        return f"{self.root / 'gpu'}:{self.path}"

    def listing(self, commands=None, tag="", module=None):
        """What the stub's scan reports; module is the image's [path, version]
        of rt_run, if it has one."""
        found = {
            "commands": LISTING if commands is None else commands,
            "from_source": ["your_header.py", "rt-run"],
            "tag": tag,
            "built": "",
            "self": module,
        }
        (self.state / "scan.json").write_text(json.dumps(found))

    def mounts(self, *extra):
        """A mount table in /proc/self/mountinfo's format: the root
        filesystem; data, other and job-tmp under the world's root as
        filesystems of their own; data/nested inside data, as network
        filesystems nest; and any extra mount points given."""
        points = ["/"] + [str(self.root / d) for d in ("data", "other", "job-tmp")]
        points += [str(self.root / "data/nested")] + [str(p) for p in extra]
        lines = [
            f"{i} 1 0:{i} / {p.replace(' ', chr(92) + '040')} rw - ext4 /dev/x{i} rw"
            for i, p in enumerate(points, 20)
        ]
        (self.root / "mountinfo").write_text("\n".join(lines) + "\n")

    def cosign(self):
        """Install the stub cosign, which finds every image signed."""
        stub = self.root / "stub" / "cosign"
        stub.write_text(COSIGN.format(python=sys.executable))
        stub.chmod(0o755)

    def unsign(self, digest):
        with open(self.state / "unsigned", "a", encoding="utf-8") as handle:
            handle.write(digest + "\n")

    def cosign_calls(self):
        path = self.state / "cosign_calls"
        if not path.exists():
            return []
        return [line.split("\t") for line in path.read_text().splitlines()]

    def built(self, date):
        (self.state / "built").write_text(date)

    def pulls(self):
        path = self.state / "pulls"
        return path.read_text().splitlines() if path.exists() else []

    def env(self, **extra):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("RT_", "SINGULARITY", "APPTAINER", "XDG_"))
        }
        env.update(HOME=str(self.home), PATH=self.path, STUB_DIR=str(self.state))
        env.update(RT_REGISTRY_API=self.registry.url)
        env.update(RT_MOUNTINFO=self.root / "mountinfo")
        for key, value in extra.items():
            if value is None:
                env.pop(key, None)  # as if never set
            else:
                env[key] = str(value)
        return env

    def run(self, cmd, cwd=None, **env):
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd or self.home),
            env=self.env(**env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        return Result(proc)

    def rt_run(self, *args, **kw):
        return self.run([sys.executable, RT_RUN] + list(args), **kw)

    def tool(self, name, *args, **kw):
        return self.run([self.bin / name] + list(args), **kw)

    def ours(self, path):
        return path.is_symlink() and os.readlink(path) == str(self.rt / "rt-run")
