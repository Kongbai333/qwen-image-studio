import json
import os
import subprocess
import sys
from pathlib import Path

from auth import gen_key, hash_password, verify_password
from qwen_client import parse_proxy_lines, parse_token_lines

ROOT = Path(__file__).resolve().parents[1]


def test_access_key_and_password_helpers():
    key = gen_key()
    assert key.startswith("QWEN-")
    assert len(key.split("-")) == 5
    assert not set(key) & set("01OI")

    digest, salt = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", digest, salt)
    assert not verify_password("wrong", digest, salt)


def test_token_and_proxy_parsing():
    tokens = parse_token_lines(
        "# comment\n"
        "primary|hf_example|main\n"
        "secondary|hf_second|backup|http://127.0.0.1:8080\n"
    )
    assert tokens[0][0:3] == ("primary", "hf_example", "main")
    assert tokens[1][3] == "http://127.0.0.1:8080"

    proxies = parse_proxy_lines("edge|http://127.0.0.1:8081|local|7\n")
    assert proxies[0][0:2] == ("edge", "http://127.0.0.1:8081")
    assert proxies[0][2] == "local"


def test_server_bootstraps_with_isolated_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    env = os.environ.copy()
    env.update(
        {
            "QWEN_DATA_DIR": str(data_dir),
            "QWEN_ADMIN_PASSWORD": "test-only-password",
            "QWEN_HF_TOKENS": "test|hf_test",
            "QWEN_PROXIES": "",
            "QWEN_SPACES": "",
        }
    )
    code = (
        "from fastapi.testclient import TestClient\n"
        "import server\n"
        "client = TestClient(server.app)\n"
        "response = client.get('/login')\n"
        "assert response.status_code == 200, response.status_code\n"
        "assert '登录' in response.text\n"
        "login = client.post('/api/auth/login', json={\n"
        "    'mode': 'admin', 'password': 'test-only-password'\n"
        "})\n"
        "assert login.status_code == 200, login.text\n"
        "print(server.DATA_ROOT)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert str(data_dir.resolve()) in result.stdout
    assert (data_dir / ".session.key").is_file()
    assert (data_dir / "channels.json").is_file()
    assert not (data_dir / "admin_password.txt").exists()
    assert not (ROOT / "tokens.txt").exists()

    channels = json.loads((data_dir / "channels.json").read_text(encoding="utf-8"))
    assert len(channels["channels"]) == 1


def test_release_tree_excludes_private_runtime_files():
    forbidden = {
        "tokens.txt",
        "proxies.txt",
        "licenses.json",
        "materials.json",
        "gallery.json",
        "jobs.json",
        "channels.json",
        ".admin.json",
        ".session.key",
        "admin_password.txt",
        "ps_accounts.json",
        "AGENTS.md",
    }
    present = {path.name for path in ROOT.iterdir()}
    assert forbidden.isdisjoint(present)
    assert not (ROOT / "outputs").exists()
    assert not (ROOT / "uploads").exists()
