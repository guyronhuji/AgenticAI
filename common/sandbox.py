"""Sandboxed execution of model-generated Python.

NEVER exec() model-written code in-process: it would share your memory, your
open files and your environment variables (API keys). Two sandboxes:

  subprocess (default)  separate process, `python -I` (isolated: no user
                        site-packages, no cwd on sys.path), hard timeout,
                        throwaway temp cwd, scrubbed environment, and on
                        Unix a memory cap via RLIMIT_AS.
                        LIMIT: does not block network access. Teaching-grade.

  docker (SANDBOX=docker) runs the same code in `python:3.12-slim` with
                        --network none and a memory limit. This IS a real
                        boundary (as real as containers are). Needs Docker.

Figures: pass `workdir=` to run in a persistent directory instead of a
throwaway one. Files the script saves there survive; image files the
script produced (.png .pdf .svg .jpg: new or modified in the workdir, or named
in a savefig call anywhere) are listed with full paths at the end of the
result and printed to the console, and with `open_figures=True` they are
opened on screen (Preview on macOS, the default viewer elsewhere). The child gets
MPLBACKEND=Agg so matplotlib never tries to open a window of its own.

The honest sentence for the lecture: the subprocess sandbox stops accidents
(infinite loops, file damage, key leakage); the container stops adversaries.
"""
from __future__ import annotations

import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

MODE = os.environ.get("SANDBOX", "subprocess")
MEM_MB = int(os.environ.get("SANDBOX_MEM_MB", "1024"))
FIG_EXTS = (".png", ".pdf", ".svg", ".jpg", ".jpeg")


def _limit_resources():  # runs in the child, before exec (Unix only)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_MB << 20, MEM_MB << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass


def _child_env(home: str) -> dict:
    """The ONLY variables the child sees. No keys, no proxies, no PYTHONPATH."""
    return {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
            "HOME": home, "MPLBACKEND": "Agg", "MPLCONFIGDIR": os.path.join(home, ".mpl")}


def _run_subprocess(code: str, timeout: int, cwd: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=cwd, env=_child_env(cwd),
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: execution exceeded {timeout}s (infinite loop?)"
    return _format(proc.returncode, proc.stdout, proc.stderr)


def _run_docker(code: str, timeout: int, cwd: str) -> str:
    if not shutil.which("docker"):
        return "ERROR: SANDBOX=docker but docker is not installed"
    pathlib.Path(cwd, "main.py").write_text(code, encoding="utf-8")
    cmd = ["docker", "run", "--rm", "--network", "none",
           f"--memory={MEM_MB}m", "--cpus=1", "--pids-limit=64",
           "-e", "MPLBACKEND=Agg",
           "-v", f"{cwd}:/work", "-w", "/work", "python:3.12-slim",
           "python", "-I", "main.py"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return f"ERROR: execution exceeded {timeout}s (infinite loop?)"
    return _format(proc.returncode, proc.stdout, proc.stderr)


def _format(rc, stdout, stderr):
    if rc != 0:
        return "ERROR:\n" + stderr.strip()[-2000:]
    out = stdout.strip()
    return out[-4000:] if out else "(no output: did the code forget to print()?)"


SAVEFIG_RE = re.compile(r"""savefig\(\s*(?:fname\s*=\s*)?['"]([^'"]+)['"]""")


def _find_figures(code: str, workdir: str, t0: float) -> list[str]:
    """Absolute paths of image files the script produced: anything new or modified
    in workdir, plus any file named in a savefig("...") call wherever it points."""
    found: dict[str, None] = {}
    for p in pathlib.Path(workdir).iterdir():
        if p.suffix.lower() in FIG_EXTS and p.stat().st_mtime >= t0 - 1:
            found[str(p.resolve())] = None
    for name in SAVEFIG_RE.findall(code):
        if name.startswith("~"):
            name = workdir + name[1:]            # the child's HOME is workdir
        q = pathlib.Path(name)
        q = q if q.is_absolute() else pathlib.Path(workdir) / q
        if q.exists() and q.stat().st_mtime >= t0 - 1:
            found[str(q.resolve())] = None
    return list(found)


def open_files(paths: list[str]) -> str | None:
    """Open files with the OS viewer: Preview on macOS, xdg-open on Linux, the
    associated app on Windows. Returns None on success, else an error message."""
    if not paths:
        return None
    try:
        sysname = platform.system()
        if sysname == "Darwin":
            r = subprocess.run(["open", "-a", "Preview", *paths], capture_output=True, text=True, timeout=15)
            if r.returncode != 0:                 # Preview refused (odd format?): default app
                r = subprocess.run(["open", *paths], capture_output=True, text=True, timeout=15)
            return None if r.returncode == 0 else (r.stderr.strip() or f"open exited {r.returncode}")
        if sysname == "Windows":
            for p in paths:
                os.startfile(p)  # type: ignore[attr-defined]
            return None
        if not shutil.which("xdg-open"):
            return "no xdg-open on this system"
        for p in paths:
            subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def run_python(code: str, timeout: int = 10, workdir: str | None = None,
               open_figures: bool = False) -> str:
    """Run untrusted Python, return stdout (or an ERROR: string the model can read).

    workdir: run there instead of a throwaway temp dir; files saved by the
             script persist. Figures the script saved are reported with their
             full paths (and printed to the console).
    open_figures: also open those images on screen.
    """
    runner = _run_docker if MODE == "docker" else _run_subprocess
    if workdir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return runner(code, timeout, tmp)
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    out = runner(code, timeout, workdir)
    figs = _find_figures(code, workdir, t0)
    if figs:
        for f in figs:
            print(f"[figure] {f}", flush=True)
        out += "\n[figures saved:\n  " + "\n  ".join(figs) + "\n]"
        if open_figures:
            err = open_files(figs)
            if err:
                print(f"[figure] could not open on screen: {err}", flush=True)
                out += f"\n[could not open on screen: {err}]"
            else:
                out += "\n[opened on screen]"
    elif SAVEFIG_RE.search(code):
        out += "\n[note: the code calls savefig but no new image file was found in " + workdir + "]"
    return out


if __name__ == "__main__":  # self-test, no model needed
    print(run_python("print(2**100)"))
    print(run_python("import time; time.sleep(30)", timeout=2))
    print(run_python("1/0"))
    print(run_python("x = 5"))
    print(run_python("import os; print(sorted(os.environ))"))
    print(run_python("x = bytearray(4 << 30)"))  # memory cap -> MemoryError on Unix
    # figures: a persistent workdir, a saved PNG, reported back (not opened here)
    print(run_python(
        "import numpy as np, matplotlib.pyplot as plt\n"
        "x = np.linspace(0, 10, 200); plt.plot(x, np.sin(x)); plt.savefig('sine.png', dpi=80)\n"
        "print('plotted', len(x), 'points')",
        timeout=60, workdir="sandbox_out/selftest"))
    # same filename again: still reported (modified, not only new)
    print(run_python("import matplotlib.pyplot as plt; plt.plot([0,1]); plt.savefig('sine.png'); print('again')",
                     timeout=60, workdir="sandbox_out/selftest"))
