# Corportal Report Automation

This implementation provides automated extraction of campaign health reports from the Corportal system using Playwright web automation.

## Setup

1. **Install Dependencies**
   ```bash
   pip install playwright python-dotenv
   python -m playwright install chromium
   ```

2. **Configure Environment**
   - The `.env` file has been created with your Corportal configuration
   - **IMPORTANT**: Replace `***REDACTED***` in the `.env` file with your actual password

3. **Initial Login Setup (One-time)**
   ```bash
   python -c "from backend.providers.corportal import ensure_session_state; ensure_session_state()"
   ```
   - This opens a browser window for one-time interactive login
   - Log in manually if auto-fill fails
   - Session state is saved to `backend/data/storage/corportal_storage_state.json`

## Usage

### Run Report Export
```bash
python backend/jobs/ingest_corportal.py
```

### Test Provider Directly
```bash
python backend/providers/corportal.py
```

## File Structure

- **`.env`** - Configuration with URLs, credentials, and paths
- **`backend/providers/corportal.py`** - Main provider with Playwright automation
- **`backend/jobs/ingest_corportal.py`** - Job wrapper for schedulers
- **`backend/data/raw/corportal/`** - Raw downloaded files
- **`backend/data/book/`** - Final files named as `YYYY-MM-DD-campaign-health.csv`
- **`backend/data/storage/`** - Browser session state
- **`backend/data/ingest_manifest.json`** - Manifest tracking all downloads with SHA256 deduplication

## Features

- **Session Management**: One-time login with persistent session state
- **Robust Form Handling**: Multiple selector strategies for login forms
- **Deduplication**: SHA256-based duplicate detection in JSON manifest
- **Error Handling**: Handles auth expiration and export timeouts
- **File Management**: Organized storage in raw/ and book/ directories
- **Timezone Support**: Configurable timezone for file naming

## Return Codes

- **0**: Success
- **2**: Authentication expired (for scheduler alerting)

## Troubleshooting

1. **Login Issues**: If auto-fill fails, manually enter credentials when the browser opens
2. **Export Timeout**: Increase timeout values in the provider if reports take longer to generate
3. **Session Expired**: Delete `corportal_storage_state.json` and re-run the login setup
4. **Path Issues**: Ensure all directories in `.env` exist and are accessible

## Security Notes

- Never commit the `.env` file with real credentials
- Session state file contains authentication cookies - protect accordingly
- Consider using environment variables instead of `.env` in production