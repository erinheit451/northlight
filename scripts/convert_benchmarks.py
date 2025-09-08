import csv, json, re, sys, hashlib, time
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / f"benchmarks_{time.strftime('%Y%m%d')}.json"
OUT = DATA / f"{date.today():%Y-%m-%d}-benchmarks.json"

MEDIANS_CSV = DATA / "Benchmark Data - Standard.csv"
DMS_CSV = DATA / "Benchmark Data - DMS-Ultimate.csv"

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def key(category: str, subcategory: str) -> str:
    return f"{norm(category)}|{norm(subcategory)}"

def load_medians(fp: Path):
    rows = {}
    with fp.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            cat = norm(row.get("Category",""))
            sub = norm(row.get("Sub Category",""))
            if not cat or not sub: 
                continue
            k = key(cat, sub)
            try:
                rows[k] = {
                    "meta": {
                        "country": norm(row.get("Country","")),
                        "currency": norm(row.get("Currency","")),
                        "category": cat,
                        "subcategory": sub,
                        "vertical": norm(row.get("Vertical","")),
                    },
                    "budget": {"median": float(row.get("Budget Median") or 0) or None},
                    "cpl":    {"median": float(row.get("Cost Per Lead Median") or 0) or None},
                    "cpc":    {"median": float(row.get("Cost Per Click Median") or 0) or None},
                    "ctr":    {"median": float(row.get("CTR Median") or 0) or None},
                    "coverage": {"has_percentiles": False, "has_ctr_median": bool(row.get("CTR Median"))}
                }
            except ValueError:
                # skip malformed numeric rows
                continue
    return rows

def _f(x):
    try:
        return float(x) if x not in (None, "",) else None
    except ValueError:
        return None

def load_dms(fp: Path):
    # Expected headers (examples):
    # BC, BSC, BSC Budget Bottom 10%, BSC Budget Bottom 25%, BSC Budget Average, BSC Budget Top 25%, BSC Budget Top 10%
    # BSC CPL Top 10%, BSC CPL Top 25%, BSC CPL Avg, BSC CPL Bottom 25%, BSC CPL Bottom 10%
    # BSC CPC Top 10%, BSC CPC Top 25%, BSC CPC Average, BSC CPC Bottom 25%, BSC CPC Bottom 10%
    rows = {}
    with fp.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            cat = norm(row.get("BC",""))
            sub = norm(row.get("BSC",""))
            if not cat or not sub:
                continue
            k = key(cat, sub)
            # budgets (note: “Top”/“Bottom” in source; keep as provided)
            budget = {
                "p10_bottom": _f(row.get("BSC Budget Bottom 10%")),
                "p25_bottom": _f(row.get("BSC Budget Bottom 25%")),
                "avg":        _f(row.get("BSC Budget Average")),
                "p25_top":    _f(row.get("BSC Budget Top 25%")),
                "p10_top":    _f(row.get("BSC Budget Top 10%")),
            }
            cpl = {
                "top10":  _f(row.get("BSC CPL Top 10%")),   # best-performing CPL (lowest)
                "top25":  _f(row.get("BSC CPL Top 25%")),
                "avg":    _f(row.get("BSC CPL Avg")),
                "bot25":  _f(row.get("BSC CPL Bottom 25%")),# worst (highest)
                "bot10":  _f(row.get("BSC CPL Bottom 10%")),
            }
            cpc = {
                "top10":  _f(row.get("BSC CPC Top 10%")),
                "top25":  _f(row.get("BSC CPC Top 25%")),
                "avg":    _f(row.get("BSC CPC Average")),
                "bot25":  _f(row.get("BSC CPC Bottom 25%")),
                "bot10":  _f(row.get("BSC CPC Bottom 10%")),
            }
            # optional extra column some exports have
            mcid_avg_cpc = _f(row.get("MCID Avg CPC"))
            rows[k] = {
                "meta": {"category": cat, "subcategory": sub},
                "budget_dms": budget,
                "cpl_dms": cpl,
                "cpc_dms": cpc,
                "mcid": {"avg_cpc": mcid_avg_cpc} if mcid_avg_cpc is not None else {},
                "coverage": {"has_percentiles": True}
            }
    return rows

def merge(dms: dict, med: dict):
    out = {}
    keys = set(dms.keys()) | set(med.keys())
    for k in keys:
        base = {"meta": {}, "budget": {}, "cpl": {}, "cpc": {}, "ctr": {}, "coverage": {}}
        if k in med:
            m = med[k]
            base["meta"].update(m["meta"])
            base["budget"].update(m.get("budget", {}))
            base["cpl"].update(m.get("cpl", {}))
            base["cpc"].update(m.get("cpc", {}))
            base["ctr"].update(m.get("ctr", {}))
            base["coverage"].update(m.get("coverage", {}))
        if k in dms:
            d = dms[k]
            # keep DMS as separate namespaces to avoid pretending they are percentiles we don’t really have
            base["meta"].update({k2:v for k2,v in d.get("meta",{}).items() if v})
            base["budget"].update({"dms": d.get("budget_dms")})
            base["cpl"].update({"dms": d.get("cpl_dms")})
            base["cpc"].update({"dms": d.get("cpc_dms")})
            base.setdefault("extra", {})["mcid"] = d.get("mcid", {})
            base["coverage"]["has_percentiles"] = True
        else:
            base["coverage"].setdefault("has_percentiles", False)
        # clean Nones
        out[k] = base
    return out

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def main():
    if not MEDIANS_CSV.exists() or not DMS_CSV.exists():
        print("Missing CSV(s). Ensure both files are in /data.", file=sys.stderr)
        sys.exit(1)
    med = load_medians(MEDIANS_CSV)
    dms = load_dms(DMS_CSV)
    combined = merge(dms, med)

    blob = json.dumps(combined, ensure_ascii=False, separators=(",",":"))
    checksum = sha256_bytes(blob.encode("utf-8"))
    payload = {"version": time.strftime("%Y-%m-%d"), "checksum": checksum, "records": combined}

    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=0))
    # keep/update alias
    if ALIAS.exists():
        ALIAS.unlink()
    ALIAS.symlink_to(OUT.name) if hasattr(ALIAS, "symlink_to") else OUT.replace(ALIAS)

    print(f"Wrote: {OUT.name}")
    print(f"Alias: {ALIAS.name}")
    print(f"Records: {len(combined)}")
    print(f"Checksum: {checksum}")

if __name__ == "__main__":
    main()
