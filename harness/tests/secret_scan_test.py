"""The scanner against sample leaks, then real blocked commits in a throwaway repo. Run from anywhere.

Samples are joined at run time so this file does not trip the scanner itself.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import secret_scan as sc  # noqa: E402

failed = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        failed.append(name)


A20 = "aB3dE6gH9jK2mN5pQ8sT"
LEAKS = {
    "GitHub token": "ghp" + "_" + A20 * 2,
    "Hugging Face token": "hf" + "_" + A20 * 2,
    "Anthropic or OpenAI key": "sk" + "-ant-" + A20,
    "AWS key": "AKIA" + "ABCDEFGHIJKLMNOP",
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "assigned secret": 'api_key' + ' = "' + A20 + '"',
    "bearer token": "Authorization: Bearer " + A20 + A20,
    "credentials in URL": "https://user:" + "hunter22" + "@host.test/x",
    "email address": "someone" + "@gmail.com",
    "home folder path": "C:" + "\\Users\\someone\\file.txt",
    "phone number": "+39 " + "333 123 4567",
}
CLEAN = [
    "import os", "weights: {path: wevisone-vl-2b}", 'token = get_token()', "hash: sha256:" + "ab12" * 16,
    "noreply" + "@anthropic.com", "1234+me" + "@users.noreply.github.com", "you" + "@example.com",
    "E:/harness-data-test", "version 1.2.3", "torch==2.11.0+cu128", "C:" + "\\Windows\\System32",
    "git+https://github.com/Tencent/WeVisOne-VL@" + "9a6bcd8", "max_new_tokens: 512",
]


def test_rules():
    for rule, text in LEAKS.items():
        hits = {r for _, r, _ in sc.scan_text(text)}
        check(f"detects {rule}", rule in hits, str(hits))
    for text in CLEAN:
        hits = sc.scan_text(text)
        check(f"clean: {text[:40]}", not hits, str(hits))
    leak = LEAKS["email address"]
    check("allow mark skips a line", not sc.scan_text(leak + "  # " + sc.ALLOW_MARK))
    check("personal term found, any case", sc.scan_text("hello SomeName", ["somename"]))
    check("findings never print the secret", all(leak not in s for _, _, s in sc.scan_text(leak)))


def test_commits():
    tmp = Path(tempfile.mkdtemp())
    repo = tmp / "repo"
    tools = repo / "harness" / "tools"
    shutil.copytree(ROOT / "tools" / "hooks", tools / "hooks")
    shutil.copy(ROOT / "tools" / "secret_scan.py", tools)
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env)

    def commit(path, text, message="add file"):
        (repo / path).write_text(text, encoding="utf-8")
        git("add", "-A")
        return git("commit", "-m", message)

    try:
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        git("config", "core.hooksPath", "harness/tools/hooks")
        git("config", "user.name", "t")
        git("config", "user.email", "1+t@users.noreply.github.com")
        check("clean commit passes", commit("a.txt", "hello\n").returncode == 0)
        r = commit("b.txt", LEAKS["GitHub token"] + "\n")
        check("key in a file blocks the commit", r.returncode != 0, r.stderr.strip()[:90])
        check("blocked output hides the key", LEAKS["GitHub token"] not in r.stderr)
        git("reset", "-q", "--hard")
        r = commit("c.txt", "fine\n", "mail me at " + LEAKS["email address"])
        check("email in the message blocks the commit", r.returncode != 0, r.stderr.strip()[:90])
        git("reset", "-q", "--hard")
        (repo / "secret-scan.local.txt").write_text("somename\n", encoding="utf-8")
        r = commit("d.txt", "by SomeName\n")
        check("local personal term blocks the commit", r.returncode != 0, r.stderr.strip()[:90])
        (repo / "secret-scan.local.txt").unlink()
        git("reset", "-q", "--hard")
        git("config", "user.email", LEAKS["email address"])
        r = commit("e.txt", "fine\n")
        check("real author email blocks the commit", r.returncode != 0, r.stderr.strip()[:90])
    finally:
        shutil.rmtree(tmp, onerror=lambda f, p, _: (os.chmod(p, stat.S_IWRITE), f(p)))


if __name__ == "__main__":
    test_rules()
    test_commits()
    print(f"\n{'all passed' if not failed else f'failed: {failed}'}")
    sys.exit(1 if failed else 0)
