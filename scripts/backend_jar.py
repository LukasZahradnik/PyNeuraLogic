#!/usr/bin/env python3
"""Put the backend jar this frontend is pinned to in place.

The jar is not tracked in this repository. What is tracked is backend.pin, naming the backend repository
and the commit this frontend belongs to, and this script turns that into neuralogic/jar/NeuraLogic.jar.
CI runs exactly this, which is the point: the pin is a recipe both a person and a workflow can execute.

    python scripts/backend_jar.py                 build the pinned commit and install it
    python scripts/backend_jar.py --jar PATH      install a jar you already built, no network, no Maven
    python scripts/backend_jar.py --check         only report whether the installed jar matches the pin

--jar is the local loop: build the backend however you like - your own clone, your own fork, an IDE run -
and hand the result over. Nothing here needs to be involved in how it was made.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = ROOT / "backend.pin"
JAR = ROOT / "neuralogic" / "jar" / "NeuraLogic.jar"
BUILT = "CLI/target/neuralogic-cli-0.3.0-jar-with-dependencies.jar"


def read_pin(path=PIN):
    """The pin is deliberately not TOML: the test matrix reaches back to Python 3.10, which has no tomllib."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    missing = {"repo", "ref"} - values.keys()
    if missing:
        raise SystemExit(f"{path} is missing {', '.join(sorted(missing))}")
    return values["repo"], values["ref"]


def stamp(jar=JAR):
    """What the jar says about itself, from the manifest the backend writes at package time."""
    if not jar.exists():
        return None
    with zipfile.ZipFile(jar) as archive:
        manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
    entries = {}
    for line in manifest.replace("\r\n", "\n").split("\n"):
        key, sep, value = line.partition(":")
        if sep and key.startswith("Backend-"):
            entries[key.strip()] = value.strip()
    return entries


def describe(jar=JAR):
    entries = stamp(jar)
    if entries is None:
        return f"no jar at {jar}"
    commit = entries.get("Backend-Commit", "unstamped")
    return f"{commit} built {entries.get('Backend-Built', 'at an unknown time')}"


def install(source, dest=JAR):
    if not zipfile.is_zipfile(source):
        raise SystemExit(f"{source} is not a jar")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    print(f"installed {dest.relative_to(ROOT)}: {describe(dest)}")


def build(repo, ref, workdir):
    """Clone or update the pinned backend and package it. Any fork, any branch, any commit."""
    workdir.mkdir(parents=True, exist_ok=True)
    clone = workdir / "NeuraLogic"
    url = repo if "://" in repo else f"https://github.com/{repo}.git"
    if not (clone / ".git").is_dir():
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", url], check=True)
    subprocess.run(["git", "-C", str(clone), "fetch", "--tags", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "--detach", "FETCH_HEAD"], check=True)

    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    mvn = os.environ.get("MAVEN", "mvn")
    subprocess.run([mvn, "-B", "-DskipTests", f"-Dbackend.commit={sha}", "package"],
                   cwd=clone, check=True, shell=os.name == "nt")
    return clone / BUILT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jar", type=pathlib.Path, help="install this jar instead of building one")
    parser.add_argument("--check", action="store_true", help="only verify the installed jar against the pin")
    parser.add_argument("--workdir", type=pathlib.Path, default=ROOT / ".backend",
                        help="where the backend clone is kept (default .backend)")
    args = parser.parse_args(argv)

    repo, ref = read_pin()

    if args.check:
        entries = stamp()
        if entries is None:
            print(f"no backend jar. Run: python scripts/backend_jar.py", file=sys.stderr)
            return 1
        commit = entries.get("Backend-Commit", "unstamped")
        if not commit.startswith(ref) and not ref.startswith(commit):
            print(f"the installed jar is not the pinned backend.\n"
                  f"  pinned:    {repo} at {ref}\n"
                  f"  installed: {commit}\n"
                  f"Run: python scripts/backend_jar.py", file=sys.stderr)
            return 1
        print(f"jar matches the pin: {describe()}")
        return 0

    install(args.jar if args.jar else build(repo, ref, args.workdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
