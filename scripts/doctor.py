# scripts/doctor.py
import os, sys, json, time, subprocess, socket, contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"

def ok(msg):  print(f"[OK]  {msg}")
def warn(msg):print(f"[WARN] {msg}")
def fail(msg):print(f"[FAIL] {msg}")

def check_python():
    v = sys.version_info
    if v < (3,10):
        fail(f"Python {v.major}.{v.minor} < 3.10")
        return False
    ok(f"Python {v.major}.{v.minor}")
    return True

def check_requirements():
    try:
        import pkg_resources
        reqs = pkg_resources.parse_requirements((PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8"))
        missing = []
        for r in reqs:
            try:
                pkg_resources.require(str(r))
            except Exception as e:
                missing.append(str(r))
        if missing:
            fail(f"Missing or incompatible: {', '.join(missing)}")
            return False
        ok("requirements satisfied")
        return True
    except Exception as e:
        warn(f"Could not verify requirements: {e}")
        return True

def check_imports_and_data():
    # ensure backend on path
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import backend.config as cfg
        p = cfg.DATA_FILE
        exists = Path(p).exists()
        if not exists:
            fail(f"DATA_FILE not found: {p}")
            return False
        ok(f"DATA_FILE exists: {p}")
    except Exception as e:
        fail(f"Import backend.config failed: {e}")
        return False

    try:
        import backend.data.loader as L
        _ = L.get_bench()
        ok("backend.data.loader imports and loads")
    except Exception as e:
        fail(f"backend.data.loader problem: {e}")
        return False

    # key services used by /diagnose
    try:
        import backend.services.analysis as A
        assert hasattr(A, "derive_inputs")
        ok("backend.services.analysis OK (derive_inputs present)")
    except Exception as e:
        fail(f"backend.services.analysis problem: {e}")
        return False

    try:
        import backend.routers.diagnose as R
        ok("backend.routers.diagnose imports")
    except Exception as e:
        fail(f"backend.routers.diagnose problem: {e}")
        return False

    return True

@contextlib.contextmanager
def uvicorn_server():
    # choose a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        host, port = s.getsockname()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    cmd = [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # wait a moment
        time.sleep(1.2)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

def http_json(url, method="GET", body=None):
    import urllib.request
    req = urllib.request.Request(url, method=method, headers={"Content-Type":"application/json"})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    else:
        data = None
    with urllib.request.urlopen(req, data=data, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("== northlight doctor ==\n")
    ok("Project root: " + str(PROJECT_ROOT))

    all_good = True
    all_good &= check_python()
    all_good &= check_requirements()
    all_good &= check_imports_and_data()

    if not all_good:
        fail("Static checks failed. Fix above and re-run.")
        sys.exit(1)

    # Spin up server and probe endpoints
    try:
        with uvicorn_server() as base:
            meta = http_json(base + "/meta")
            ok(f"/meta ok (data_version={meta.get('data_version')})")

            bm = http_json(base + "/benchmarks/meta")
            ok(f"/benchmarks/meta ok (items={len(bm)})")

            # sample diagnose call (tweak category/subcategory to ones that exist in your data)
            body = {
                "website": None,
                "category": bm[0]["category"],
                "subcategory": bm[0]["subcategory"],
                "budget": 1000.0,
                "clicks": 500,
                "leads": 25,
                "goal_cpl": 40.0,
                "impressions": 20000,
                "dash_enabled": True
            }
            diag = http_json(base + "/diagnose", method="POST", body=body)
            ok(f"/diagnose ok (key={diag['meta']['category_key']})")
    except Exception as e:
        fail(f"Live API check failed: {e}")
        sys.exit(2)

    print("\nAll checks passed ✅")

if __name__ == "__main__":
    main()
