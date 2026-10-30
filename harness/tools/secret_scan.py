"""Block API keys, emails and personal data before they reach git. Standard library only.

    secret_scan.py                  staged files + commit identity (pre-commit hook)
    secret_scan.py --message FILE   a commit message (commit-msg hook)
    secret_scan.py --all            every tracked file (CI)
    secret_scan.py --history        every email in the commit history

Personal terms that must never be published go in secret-scan.local.txt at the repo root, one per line. That
file is ignored by git. A false positive can be allowed with the text "secret-scan: allow" on the same line.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARK = "secret-scan: allow"
LOCAL_TERMS = "secret-scan.local.txt"
SAFE_EMAIL = re.compile(r"@users\.noreply\.github\.com$|^no-?reply@|@example\.(com|org|net)$", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

RULES = [
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("Anthropic or OpenAI key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("AWS key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google key", re.compile(r"\bAIza[A-Za-z0-9_-]{35}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned secret", re.compile(
        r"(?i)\b[\w.-]*(?:api[_-]?key|secret|token|passwd|password)[\w.-]*[\"']?\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}")),
    ("credentials in URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")),
    ("home folder path", re.compile(r"(?i)(?:[A-Z]:[\\/]+Users[\\/]+|/home/|/Users/)(?!public\b|default\b|runner\b)[\w.-]+")),
    ("phone number", re.compile(r"(?<![\w.])\+\d{2,3}[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b")),
]


def scan_text(text: str, terms: list[str] = ()) -> list[tuple[int, str, str]]:
    """Findings as (line number, rule, the match with most of it masked)."""
    found = []
    lowered = [t.lower() for t in terms]
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line:
            continue
        for name, rule in RULES:
            for m in rule.finditer(line):
                found.append((n, name, mask(m.group())))
        for m in EMAIL.finditer(line):
            if not SAFE_EMAIL.search(m.group()):
                found.append((n, "email address", mask(m.group())))
        low = line.lower()
        for term in lowered:
            if term in low:
                found.append((n, "personal term", mask(term)))
    return found


def mask(value: str) -> str:
    return value[:4] + "…" if len(value) > 4 else "…"


def git(*args: str, check: bool = True) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=check).stdout


def local_terms(root: Path) -> list[str]:
    path = root / LOCAL_TERMS
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [t.strip() for t in lines if t.strip() and not t.startswith("#")]


def scan_paths(paths: list[str], blob, terms: list[str]) -> list[str]:
    errors = []
    for path in paths:
        if path == LOCAL_TERMS:
            errors.append(f"{path}: this local file must never be committed")
            continue
        # latin-1 keeps every byte, so text inside binaries such as PNG metadata is scanned too
        for n, rule, shown in scan_text(blob(path).decode("latin-1"), terms):
            errors.append(f"{path}:{n}: {rule}: {shown}")
    return errors


def identity_errors() -> list[str]:
    errors = []
    for who in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        ident = git("var", who).decode(errors="replace")
        email = ident[ident.find("<") + 1:ident.find(">")]
        if not SAFE_EMAIL.search(email):
            errors.append(f"{who}: uses a real email address ({mask(email)}). "
                          "Set your GitHub noreply address with: git config user.email ...")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(prog="secret_scan")
    ap.add_argument("--message", type=Path, help="scan a commit message file")
    ap.add_argument("--all", action="store_true", help="scan every tracked file")
    ap.add_argument("--history", action="store_true", help="check every email in the commit history")
    a = ap.parse_args()

    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    terms = local_terms(root)
    if a.message:
        lines = [l for l in a.message.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
        errors = [f"commit message:{n}: {rule}: {shown}" for n, rule, shown in scan_text("\n".join(lines), terms)]
    elif a.history:
        emails = set(git("log", "--all", "--format=%ae%n%ce").decode(errors="replace").split())
        errors = [f"history: {mask(e)}" for e in sorted(emails) if not SAFE_EMAIL.search(e)]
    elif a.all:
        paths = git("ls-files", "-z").decode().split("\0")[:-1]
        errors = scan_paths(paths, lambda p: (root / p).read_bytes() if (root / p).is_file() else b"", terms)
    else:
        staged = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")[:-1]
        errors = identity_errors() + scan_paths(staged, lambda p: git("show", f":{p}"), terms)

    if errors:
        print("secret scan failed:\n  " + "\n  ".join(errors), file=sys.stderr)
        print(f'Remove these, or add "{ALLOW_MARK}" to a line that is a false positive.', file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
