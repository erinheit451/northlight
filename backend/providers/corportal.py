# -*- coding: utf-8 -*-
from __future__ import annotations
import os, hashlib, json, time, pathlib, shutil
from datetime import datetime
from urllib.parse import urljoin
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

load_dotenv()

LOGIN_URL = os.getenv("CORPORTAL_LOGIN_URL").strip()
REPORT_URL = os.getenv("CORPORTAL_REPORT_URL").strip()
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT", "data")).expanduser()
RAW_DIR = DATA_ROOT / "raw" / "corportal"
STATE_FILE = DATA_ROOT / "storage" / "corportal_storage_state.json"
MANIFEST = DATA_ROOT / "ingest_manifest.json"
BOOK_DIR = pathlib.Path(os.getenv("BOOK_DIR", "")).expanduser()
TZ = os.getenv("TIMEZONE", "America/Los_Angeles")

USER = os.getenv("CORPORTAL_USERNAME")
PWD  = os.getenv("CORPORTAL_PASSWORD")

RAW_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
BOOK_DIR.mkdir(parents=True, exist_ok=True)

def _now_ts_pt():
    # We'll use local machine time; if you require strict PT, run the job on a PT clock.
    return datetime.now().strftime("%Y-%m-%d")

def _load_manifest() -> dict:
    if MANIFEST.exists():
        try: return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except: return {"files": []}
    return {"files": []}

def _save_manifest(obj: dict) -> None:
    MANIFEST.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _headful_login_and_save_state():
    # One-time interactive login to capture session
    print("Opening headful browser for one-time login…")
    with sync_playwright() as p:
        br = p.chromium.launch(headless=False)
        ctx = br.new_context()
        page = ctx.new_page()
        page.goto(LOGIN_URL, timeout=90_000)
        
        # Fill in the login form - based on actual HTML structure
        try:
            # Email field name is "ux" 
            page.locator('input[name="ux"]').fill(USER, timeout=10000)
            print("Filled email field")
            
            # Password field name is "px"
            page.locator('input[name="px"]').fill(PWD, timeout=10000)
            print("Filled password field")
            
        except Exception as e:
            print(f"Auto-fill failed: {e}")
            print("Please manually fill in credentials and submit the form...")
            print("The script will continue once you successfully log in.")
        
        # Click the Login button
        try:
            # The submit button has name="commit" and value="Login"
            page.locator('input[name="commit"]').click(timeout=10000)
            print("Clicked login button")
                
        except Exception as e:
            print(f"Auto-submit failed: {e}")
            print("Please manually click the login button...")
        
        # Wait for post-login redirect - look for any page that's not the login page
        print("Waiting for successful login...")
        try:
            # Wait for URL to change away from login page
            page.wait_for_url(lambda url: "logon.php" not in url, timeout=60_000)
            print("Login successful - redirected away from login page")
        except Exception as e:
            print(f"Navigation wait failed: {e}")
            # Try to continue anyway if we're not on login page
            if "logon.php" not in page.url:
                print("Already past login page, continuing...")
            else:
                print("Still on login page - login may have failed")
        
        # Navigate to the report URL to test access
        print("Navigating to report URL to verify access...")
        try:
            page.goto(REPORT_URL, timeout=60_000)
            print("Successfully accessed report page")
        except Exception as e:
            print(f"Failed to access report page: {e}")
        
        # Save session state
        ctx.storage_state(path=str(STATE_FILE))
        print(f"Saved session to {STATE_FILE}")
        br.close()

def ensure_session_state():
    if not STATE_FILE.exists():
        _headful_login_and_save_state()

def export_report_once() -> dict:
    """
    Navigates to the report and clicks the Export link (excel_version=1).
    Saves the file into RAW_DIR; then copies/renames into BOOK_DIR as YYYY-MM-DD-campaign-health.csv
    Deduped via sha256 tracked in JSON manifest.
    """
    ensure_session_state()
    summary = {"status":"ok", "raw_path": None, "book_path": None, "sha256": None, "duplicate": False}
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True)
        ctx = br.new_context(storage_state=str(STATE_FILE), accept_downloads=True)
        page = ctx.new_page()

        # If session expired, we'll get bounced back to login — handle that.
        page.goto(REPORT_URL, timeout=120_000, wait_until="domcontentloaded")
        if "logon.php" in page.url.lower() or "login" in page.url.lower():
            br.close()
            summary["status"] = "AUTH_EXPIRED"
            return summary

        # Try clicking Export link by href contains excel_version=1
        export_link = None
        try:
            export_link = page.locator("a[href*='excel_version=1']")
            if export_link.count() == 0:
                # Fallback: link text "Export"
                export_link = page.get_by_role("link", name=lambda n: "export" in n.lower())
            # Scroll into view; some pages load link at bottom
            export_link.first.scroll_into_view_if_needed()
            with page.expect_download(timeout=120_000) as dl_info:
                export_link.first.click()
            download = dl_info.value
        except PWTimeout:
            br.close()
            summary["status"] = "EXPORT_TIMEOUT"
            return summary

        # Save into a temp file in RAW_DIR
        ts = int(time.time())
        temp_path = RAW_DIR / f"export_{ts}"
        download.save_as(temp_path)
        # Normalize extension: if no extension, try to infer
        guessed_ext = (download.suggested_filename.split(".")[-1].lower()
                       if "." in download.suggested_filename else "")
        final_raw = RAW_DIR / (download.suggested_filename if guessed_ext else f"export_{ts}.csv")
        shutil.move(str(temp_path), str(final_raw))
        br.close()

    # Compute sha and dedupe
    sha = _sha256(final_raw)
    summary["raw_path"] = str(final_raw)
    summary["sha256"] = sha

    manifest = _load_manifest()
    seen = any(row.get("sha256") == sha for row in manifest.get("files", []))
    if seen:
        summary["duplicate"] = True
        # Keep the raw for audit, but DO NOT copy to book again.
    else:
        # Name for book dir: YYYY-MM-DD-campaign-health.csv in PT
        day = _now_ts_pt()
        book_name = f"{day}-campaign-health.csv"
        book_path = BOOK_DIR / book_name
        # If a same-day file exists, we'll overwrite with the latest export (that's usually desirable)
        shutil.copyfile(final_raw, book_path)
        summary["book_path"] = str(book_path)

    # Update manifest
    manifest.setdefault("files", []).append({
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "report_url": REPORT_URL,
        "raw_path": str(final_raw),
        "book_path": summary["book_path"],
        "sha256": sha,
        "duplicate": summary["duplicate"],
        "status": summary["status"],
        "suggested_filename": download.suggested_filename if 'download' in locals() else None
    })
    _save_manifest(manifest)
    return summary

if __name__ == "__main__":
    out = export_report_once()
    print(json.dumps(out, indent=2))