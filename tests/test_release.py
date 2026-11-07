"""The release check must catch what the brief forbids, and the allowlist must not let it through."""
from delayed_local_cp.release import build, check

MAIL = "someone@" + "iau.ir"
KEY = "-----BEGIN RSA PRIVATE " + "KEY-----"
SPEC = {"include": ["src/"], "forbidden": ["runs/", ".env", "paper/", "*.pdf"], "max_file_bytes": 100, "allowed_email_files": ["CITATION.cff"]}


def _repo(tmp_path):
    out = tmp_path / "out"
    (out / ".git").mkdir(parents=True)
    return out


def test_clean_tree_passes(tmp_path):
    out = _repo(tmp_path)
    (out / "a.py").write_text("x = 1\n")
    (out / "CITATION.cff").write_text(f"email: {MAIL}\n")
    assert check(out, SPEC) == []


def test_each_rule_fires(tmp_path):
    out = _repo(tmp_path)
    (out / "runs").mkdir()
    (out / "runs" / "r.json").write_text("{}")
    (out / ".env").write_text("A=1\n")
    (out / "m.pdf").write_bytes(b"\xff\xfe\x00")
    (out / "big.bin").write_bytes(b"\xff" * 200)
    (out / "key.txt").write_text(KEY + "\n")
    (out / "path.txt").write_text("see " + "/mnt" + "/storage/x\n")
    (out / "mail.txt").write_text(f"ask {MAIL}\n")
    joined = "\n".join(check(out, SPEC))
    for expected in ("forbidden path: runs/r.json", "forbidden path: .env", "forbidden path: m.pdf", "over 100 bytes: big.bin", "secret: key.txt", "host path", "e-mail address"):
        assert expected in joined, expected


def test_build_copies_only_the_allowlist_and_drops_stale_files(tmp_path):
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text("x = 1\n")
    (root / "paper").mkdir()
    (root / "paper" / "main.tex").write_text("secret draft\n")
    (root / "results" / "summaries").mkdir(parents=True)
    out = _repo(tmp_path)
    (out / "stale.txt").write_text("old\n")
    build(root, out, SPEC, "abc", "tag")
    assert (out / "src" / "m.py").exists() and not (out / "paper").exists() and not (out / "stale.txt").exists()
    assert (out / ".git").is_dir() and "abc" in (out / "RELEASE.md").read_text()
