"""The jar in the tree has to be the backend this frontend says it belongs to.

Without this the mismatch still gets caught - by whichever test happens to call the missing thing first,
reporting a frontend file and an attribute that is not there. That has happened, and cost a release. Here
it is one assertion that names both sides.
"""

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("backend_jar", ROOT / "scripts" / "backend_jar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_jar_in_the_tree_is_the_pinned_backend():
    backend_jar = _script()
    repo, ref = backend_jar.read_pin()
    entries = backend_jar.stamp()

    assert entries is not None, (
        f"no jar at {backend_jar.JAR}. It is not tracked - build it with scripts/backend_jar.py"
    )

    commit = entries.get("Backend-Commit", "unknown")
    assert commit != "unknown", (
        "the jar carries no commit stamp, so it was built outside the recipe and there is no way to tell "
        f"whether it is {repo} at {ref}"
    )
    if not backend_jar.names_a_commit(ref):
        # A branch pin is for work in flight on both sides at once. Which commit its tip is right now
        # is a question for the network, and a test suite is the wrong place to ask one - so this
        # asserts what can be known offline and says so rather than asserting something weaker in
        # silence.
        return

    assert commit.startswith(ref) or ref.startswith(commit), (
        f"the jar is not the pinned backend: pinned {repo} at {ref}, installed {commit}"
    )


def test_a_missing_jar_is_reported_as_itself(tmp_path):
    """The cost of not tracking the jar is that a fresh checkout has none, and that has to read as such.

    In a fresh interpreter, deliberately: initialize() refuses a second call while a JVM is live, so
    in-process this would report "already initialized" and never reach the jar - which is exactly the
    shape of mistake that makes a test pass without testing anything.
    """
    program = """
import neuralogic
try:
    neuralogic.initialize(jar_path=r'{absent}')
except FileNotFoundError as error:
    print("RAISED")
    print(error)
""".format(absent=tmp_path / "absent.jar")

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, cwd=ROOT)

    assert "RAISED" in result.stdout, f"initialize did not refuse a missing jar: {result.stdout}{result.stderr}"
    assert "no backend jar" in result.stdout
