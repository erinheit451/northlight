# Northlight Directory Structure

```
C:\Users\Roci\northlight\
├── 📄 .env                              # Environment configuration (Corportal credentials)
├── 📄 run_corportal_report.bat          # Scheduled task batch file
├── 📄 CORPORTAL_README.md               # Corportal setup documentation
├── 📄 main.py                           # Main application entry point
├── 📄 requirements.txt                  # Python dependencies
├── 📄 Procfile                          # Deployment configuration
│
├── 📁 backend/                          # Backend Python application
│   ├── 📄 __init__.py
│   ├── 📄 config.py                     # Application configuration
│   │
│   ├── 📁 analytics/                    # Analytics modules
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 book/                         # Data processing and playbooks
│   │   ├── 📄 __init__.py
│   │   ├── 📄 ingest.py                 # Data ingestion logic
│   │   ├── 📄 partners_cache.py         # Partner data caching
│   │   ├── 📄 partners_data.py          # Partner data processing
│   │   ├── 📄 state.py                  # State management
│   │   └── 📁 playbooks/                # Campaign playbooks
│   │       ├── 📄 __init__.py
│   │       ├── 📄 registry.py           # Playbook registry
│   │       ├── 📄 types.py              # Type definitions
│   │       └── 📄 seo_dash.yaml         # SEO dashboard config
│   │
│   ├── 📁 data/                         # Data storage and processing
│   │   ├── 📄 __init__.py
│   │   ├── 📄 campaign_loader.py        # Campaign data loader
│   │   ├── 📄 loader.py                 # General data loader
│   │   ├── 📄 snapshots.py              # Data snapshots
│   │   ├── 📄 ingest_manifest.json      # 🆕 Corportal download tracking
│   │   ├── 📁 book/                     # Processed campaign data
│   │   │   ├── 📄 2025-09-11-campaign-health.csv  # 🆕 Daily reports
│   │   │   ├── 📄 2025-09-08-campaign-health.csv
│   │   │   └── 📄 state.json
│   │   ├── 📁 raw/                      # Raw data storage
│   │   │   └── 📁 corportal/            # 🆕 Corportal raw downloads
│   │   │       └── 📄 Corporate_Portal_run.csv
│   │   └── 📁 storage/                  # 🆕 Browser session storage
│   │       └── 📄 corportal_storage_state.json
│   │
│   ├── 📁 exporters/                    # Data export utilities
│   │   ├── 📄 __init__.py
│   │   └── 📄 ppt.py                    # PowerPoint exporter
│   │
│   ├── 📁 jobs/                         # 🆕 Scheduled job scripts
│   │   └── 📄 ingest_corportal.py       # 🆕 Daily Corportal extraction job
│   │
│   ├── 📁 models/                       # Data models
│   │   ├── 📄 __init__.py
│   │   └── 📄 io.py                     # Input/output models
│   │
│   ├── 📁 policy/                       # Business logic policies
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 providers/                    # 🆕 External data providers
│   │   └── 📄 corportal.py              # 🆕 Corportal automation provider
│   │
│   ├── 📁 routers/                      # API endpoints
│   │   ├── 📄 __init__.py
│   │   ├── 📄 diagnose.py               # Diagnosis endpoints
│   │   └── 📄 export.py                 # Export endpoints
│   │
│   ├── 📁 sales/                        # Sales-related modules
│   │
│   ├── 📁 scoring/                      # Scoring algorithms
│   │
│   ├── 📁 services/                     # Business services
│   │   ├── 📄 __init__.py
│   │   ├── 📄 analysis.py               # Analysis services
│   │   ├── 📄 diagnosis.py              # Diagnosis services
│   │   ├── 📄 goal.py                   # Goal management
│   │   └── 📄 projections.py            # Projection calculations
│   │
│   └── 📁 utils/                        # Utility functions
│       ├── 📄 __init__.py
│       └── 📄 math.py                   # Mathematical utilities
│
├── 📁 frontend/                         # Frontend web application
│   ├── 📄 index.html                    # Main web interface
│   ├── 📄 styles.css                    # Styling
│   ├── 📄 script.js                     # JavaScript logic
│   ├── 📄 favicon.ico                   # Site icon
│   │
│   ├── 📁 book/                         # Book-specific frontend
│   │   ├── 📄 test_partners.html        # Partner testing interface
│   │   └── 📄 risk_waterfall.js         # Risk visualization
│   │
│   ├── 📁 credit/                       # Credit-related frontend
│   ├── 📁 growth/                       # Growth-related frontend
│   └── 📁 northlight-favicon-pack/      # Favicon assets
│
├── 📁 scripts/                          # Utility scripts
│   ├── 📄 doctor.py                     # System diagnostics
│   ├── 📄 doctor.ps1                    # PowerShell diagnostics
│   └── 📄 book_smoke.py                 # Smoke testing
│
└── 📁 tests/                            # Test suite
    └── 📄 test_churn_model.py           # Churn model tests

## 🆕 New Corportal Integration Files

- **`.env`** - Contains Corportal login credentials and configuration
- **`backend/providers/corportal.py`** - Playwright automation for report extraction  
- **`backend/jobs/ingest_corportal.py`** - Scheduled job wrapper
- **`backend/data/storage/corportal_storage_state.json`** - Browser session state
- **`backend/data/raw/corportal/`** - Raw downloaded files storage
- **`backend/data/book/*.csv`** - Processed daily campaign health reports
- **`backend/data/ingest_manifest.json`** - Download tracking with deduplication
- **`run_corportal_report.bat`** - Windows Task Scheduler batch file

## Windows Task Scheduler

**Task Name:** "Corportal Report Extraction"  
**Schedule:** Daily at 7:00 AM  
**Action:** Runs `run_corportal_report.bat`
```