#!/usr/bin/env python3
"""Put the backend jar named in backend.pin in place.

The jar is not tracked in this repository. backend.pin names the backend repository and the ref this
frontend belongs to, and this script turns that into neuralogic/jar/NeuraLogic.jar. CI runs the same
script, which is the whole point: the pin is one recipe, executed identically by a person and by a
workflow, so what CI tested is what you can reproduce.

    backend_jar.py                clone or update the pinned repository, build that ref, install it
    backend_jar.py --jar PATH     install a jar you already built; no network, no Maven
    backend_jar.py --check        report whether the installed jar is the pinned one

--jar is the local loop: build the backend however you like - your own clone, an IDE run - and hand
the result over. Nothing here needs to know how it was made.
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


def names_a_commit(ref):
    """A commit can be checked against a jar stamp without asking anybody. A branch name cannot."""
    return len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower())


def install(source, dest=JAR):
    if not zipfile.is_zipfile(source):
        raise SystemExit(f"{source} is not a jar")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    print(f"installed {dest.relative_to(ROOT)}: {describe(dest)}")


def build(repo, ref, workdir):
    """Fetch the pinned ref and package it.

    One path covers a commit, a branch and a tag: git fetch takes any of the three and leaves it at
    FETCH_HEAD, which is then checked out detached. The commit stamped into the jar is read back from
    the checkout afterwards, not taken from the pin - so a branch pin still yields a jar that names the
    exact commit it was built from, which is what --check reads later.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    clone = workdir / "NeuraLogic"
    # A local clone is how a branch that is pushed nowhere gets built at all.
    local = pathlib.Path(repo).expanduser()
    if "://" in repo:
        url = repo
    elif local.is_dir():
        url = str(local.resolve())
    else:
        url = f"https://github.com/{repo}.git"
    if not (clone / ".git").is_dir():
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "remote", "set-url", "origin", url], check=True)
    subprocess.run(["git", "-C", str(clone), "fetch", "--tags", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "--detach", "FETCH_HEAD"], check=True)

    sha = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    # Resolved rather than run through a shell, so a Maven path with spaces survives.
    mvn = os.environ.get("MAVEN") or shutil.which("mvn")
    if mvn is None:
        raise SystemExit("no mvn on PATH. Set MAVEN, or install one you already build the backend with")
    subprocess.run([mvn, "-B", "-DskipTests", f"-Dbackend.commit={sha}", "package"],
                   cwd=clone, check=True)
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
        if commit in ("unstamped", "unknown"):
            print("the installed jar carries no commit stamp: it was built either outside this "
                  "recipe, or from a backend older than the commit that started stamping them. "
                  f"Either way there is no telling whether it is {repo} at {ref}.\n"
                  "Run: python scripts/backend_jar.py", file=sys.stderr)
            return 1

        if not names_a_commit(ref):
            # Nothing here can tell a current branch tip from a stale one without asking the remote,
            # and this check is meant to run offline. Report, and leave the claim unmade.
            print(f"the pin names a branch, so this is as far as an offline check goes:\n"
                  f"  pinned:    {repo} at {ref}\n"
                  f"  installed: {commit}\n"
                  f"Whether that is the current tip of {ref} cannot be told from here.")
            return 0

        if not commit.startswith(ref) and not ref.startswith(commit):
            print(f"the installed jar is not the pinned backend.\n"
                  f"  pinned:    {repo} at {ref}\n"
                  f"  installed: {commit}\n"
                  "Run: python scripts/backend_jar.py", file=sys.stderr)
            return 1

        print(f"jar matches the pin: {describe()}")
        return 0

    install(args.jar if args.jar else build(repo, ref, args.workdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
