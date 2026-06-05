"""
Gemini-powered historical trend validator for BR-MTGNN.

Purpose
-------
Before training or plotting, audit each PT (Pertinent Technology) column in
data/data.csv for two classes of errors identified in the forecasting plots:

  1. Anachronistic values — non-zero activity recorded before the PT concept
     was coined or before it could plausibly have mainstream measurable interest.
  2. Duplicate / cross-contamination — identical value sequences across columns
     that should be independent (data template reuse).

For each flagged PT the validator zeroes-out the values in the pre-existence
window so the downstream smoothing + model training + plotting pipeline sees
historically plausible data.

All corrections are determined exclusively by Gemini with Google Search
grounding — there are no hard-coded override values. Every decision is
documented in the audit report with concise evidence, a rationale, and a
cited source.

Usage (standalone audit)
------------------------
  python gemini_validator.py \
      --input_csv data/data.csv \
      --output_csv data/data_validated.csv \
      --api_key  YOUR_GEMINI_API_KEY          # or set GEMINI_API_KEY env var
      --dry_run                               # print audit only, don't write

Integration (called from smoothing.py or forecast.py)
------------------------------------------------------
  from gemini_validator import validate_and_correct
  corrected_df, report = validate_and_correct(df, api_key=api_key)

The returned `report` is a list of dicts describing every correction applied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeminiAuthenticationError(RuntimeError):
    """Raised when Gemini rejects the configured API key."""


def _is_gemini_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "api key expired",
            "api_key_invalid",
            "api key not valid",
            "permission_denied",
            "unauthenticated",
        )
    )


def _is_gemini_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(m in text for m in ("quota", "rate_limit", "resource_exhausted", "429"))


def _is_gemini_env_error(exc: Exception) -> bool:
    """Detect environment/dependency errors that make Gemini permanently unusable."""
    text = str(exc).lower()
    return any(m in text for m in (
        "pydantic",
        "incompatible",
        "importerror",
        "modulenotfounderror",
        "cannot import",
    )) or isinstance(exc, (ImportError, ModuleNotFoundError))

# Auto-load .env from the repo root (or any parent directory) so
# GEMINI_API_KEY is available without manual export steps.
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        # Walk up from this file's directory looking for .env
        here = Path(__file__).resolve().parent
        for directory in [here, *here.parents]:
            env_file = directory / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=True)
                return
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

_load_dotenv()

# Duplicate detection threshold: correlation >= this value flags two columns as
# potentially sharing the same underlying data series.
_DUP_CORR_THRESHOLD = 0.9999

# Default cache file location — sibling to this script (repo root / data/).
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "data" / "validation_cache.json"


# Prompt/cache version. Increment when hard constraints or clinical audit rules change.
_VALIDATION_SCHEMA_VERSION = "clinical_blacklist_v3"

# RMD-to-PT combinations that are clinically implausible or repeatedly misattributed.
# The CSV stores RMD/PT names as prefixed columns. Matching is performed on
# cleaned display names such as Stendhal Syndrome and Clozapine Protocol.
CLINICAL_BLACKLIST = {
    "Stendhal Syndrome": ["Clozapine Protocol", "Antipsychotic Medications", "Electroconvulsive Therapy"],
    "Koro": ["Clozapine Protocol", "Digital Therapeutics", "Blockchain For Data Protection"],
    "Wendigo Psychosis": ["Clozapine Protocol", "Digital Therapeutics", "Blockchain For Data Protection"],
    "Factitious Disorder": ["Digital Therapeutics", "Blockchain For Data Protection"],
    "Body Integrity Dysphoria": ["Clozapine Protocol", "Antipsychotic Medications"],
    "Savant Syndrome": ["Clozapine Protocol", "Electroconvulsive Therapy"],
    "Hypergraphia": ["Antipsychotic Medications"],
}

# PT columns that are considered global data noise in this dataset and should
# never reach Gemini, smoothing, training, or plots.
CLINICAL_GLOBAL_PT_BLACKLIST = {
    "Clozapine Protocol": "Presence of Clozapine is medically inappropriate in this dataset and represents data noise.",
}

SYSTEM_INSTRUCTION = """
You are a Senior Clinical Auditor and Medical Data Scientist. Your goal is to
identify and zero-out implausible data in a rare mental disease dataset.

CRITICAL GUIDELINES:
1. MEDICAL PLAUSIBILITY:
   - Transient or culture-bound syndromes, for example Stendhal Syndrome, Koro,
     and Wendigo Psychosis, do not use high-potency antipsychotic protocols such
     as Clozapine as routine evidence-based treatment.
   - If high-intensity psychiatric medication signals such as Antipsychotics,
     Clozapine, or Lithium appear linked to transient or culture-bound syndromes,
     treat that as a likely data error unless strong clinical evidence says otherwise.
   - Condition-treatment pairing must be supported by evidence-based medicine.

2. TEMPORAL ANACRONISM:
   - Metaverse, Blockchain For Data Protection, and Digital Therapeutics are
     recent technology categories. Non-zero values before their real-world
     clinical adoption window should be treated as hallucinated or noisy history.
   - Use Google Search grounding to verify the adoption year and set
     correction_action = zero_before_year when pre-adoption activity appears.

3. DATA HYGIENE:
   - If a PT shows a massive, sudden spike that does not align with a real-world
     technological breakthrough, guideline, product launch, or clinical trial,
     mark magnitude_issue = true.
   - If a PT sequence is identical or near-identical to another PT or an RMD, it
     is likely copy-paste or template contamination and should be flagged.
""".strip()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _compute_data_hash(df: pd.DataFrame, pt_cols: List[str]) -> str:
    """
    SHA-256 of the PT column values only.
    If the raw data changes (new rows, corrected values, added columns) the hash
    changes and the cache is invalidated automatically.
    """
    numeric = df[pt_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw = numeric.values.tobytes()
    col_bytes = "||".join(pt_cols).encode()
    return hashlib.sha256(raw + col_bytes + _VALIDATION_SCHEMA_VERSION.encode()).hexdigest()


def _load_cache(cache_path: Path, data_hash: str) -> Optional[Dict[str, dict]]:
    """Return cached audit_results if the cache exists and the data hash matches."""
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
        if payload.get("data_hash") != data_hash:
            return None
        results = payload.get("audit_results")
        if isinstance(results, dict):
            return results
    except Exception:
        pass
    return None


def _save_cache(cache_path: Path, data_hash: str, audit_results: Dict[str, dict]) -> None:
    """Persist Gemini audit results to disk so future runs skip the API."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "data_hash": data_hash,
            "audit_results": audit_results,
        }, indent=2))
    except Exception as exc:
        logger.warning(f"Could not write validation cache: {exc}")


# ---------------------------------------------------------------------------
# Gemini API helper
# ---------------------------------------------------------------------------

def _gemini_chat(prompt: str, api_key: str, model: str = "gemini-2.5-flash",
                 use_grounding: bool = True) -> str:
    """
    Send a prompt to the Gemini API and return the text response.

    use_grounding=True activates Google Search grounding so Gemini can look up
    real publication dates, PubMed records, and news timelines before answering.
    Only supported on models that have the grounding capability (2.0+).
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        client = genai.Client(api_key=api_key)

        if use_grounding:
            # Enable Google Search grounding — Gemini will search the web before
            # answering, giving it access to real publication/coinage dates.
            try:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return response.text.strip()
            except Exception:
                # Grounding not available on this model/tier — fall through
                pass

        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except ImportError:
        pass

    # Fallback: legacy google-generativeai SDK (deprecated but still functional)
    try:
        import google.generativeai as genai_legacy  # type: ignore
        genai_legacy.configure(api_key=api_key)
        legacy_client = genai_legacy.GenerativeModel(model)
        response = legacy_client.generate_content(prompt)
        return response.text.strip()
    except ImportError as exc:
        raise ImportError(
            "Neither google-genai nor google-generativeai is installed. "
            "Run: pip install google-genai"
        ) from exc


def _build_master_audit_prompt(pt_entries: List[dict], dataset_start: int = 2004,
                               dataset_end: int = 2025) -> str:
    """
    Master audit prompt — replaces all previous separate prompts.

    Design principles applied
    -------------------------
    1. Role + goal framing   : Sets Gemini up as a domain expert with a specific
                               scientific mission, not a generic assistant.
    2. Web search instruction : Explicitly tells Gemini to use its Google Search
                               grounding to look up real publication/coinage dates
                               before answering — not to rely on memory alone.
    3. Rationale field       : Requires Gemini to fill a concise "reasoning"
                               field before giving its verdict, forcing
                               evidence-based analysis rather than
                               pattern-matched guessing.
    4. Confidence gating     : Requires an explicit confidence level; low-confidence
                               items are NOT auto-corrected by the calling code.
    5. Three checks in one   : Anachronism, data spike plausibility, and duplicate
                               detection all happen in a single call per batch,
                               halving the number of API calls vs. the old approach.
    6. Structured JSON schema: Every field is named, typed, and exemplified so the
                               parser never needs to guess.
    7. Failure mode guidance : Tells Gemini what to do when it is genuinely uncertain
                               (output confidence=low, do not invent a year).

    Each entry in pt_entries:
      {
        "name":         "Digital Phenotyping",       # clean readable name
        "column":       "PT_Digital Phenotyping_NoM", # raw CSV column
        "annual_means": {"2004": 3.1, ..., "2025": 9.4},
        "max_value":    47.0,                         # monthly peak across all years
        "nonzero_years": ["2004","2005",...]           # years with mean > 0
      }
    """
    entries_json = json.dumps(pt_entries, indent=2)

    return f"""{SYSTEM_INSTRUCTION}

You are a senior medical informatics scientist and technology historian \
specialising in mental health research data quality.

## YOUR MISSION
A time-series dataset covers {dataset_start}–{dataset_end} (monthly resolution) for \
{len(pt_entries)} Pertinent Technologies (PTs) linked to rare mental disorders. \
You must audit EVERY PT for three classes of data error, then produce a correction \
plan that will be automatically applied to the dataset before model training.

## IMPORTANT — USE GOOGLE SEARCH
Before answering each PT, search the internet to verify:
- The exact year the term/technology was coined or first appeared in peer-reviewed literature
- The year it first showed measurable public interest (Google Trends spike, PubMed volume, \
clinical guideline adoption)
- Whether the values in the data are plausible in magnitude for a rare-disease research \
context (typical monthly count range: 0–200)

Do NOT rely solely on memory. Use search to verify uncertain dates.

## THREE CHECKS TO PERFORM FOR EACH PT

### Check A — Anachronism (most critical)
Determine the REAL-WORLD "first_valid_year": the earliest year the technology could \
plausibly generate non-zero monthly counts in a health research database.
- If the concept was well-established before {dataset_start}: first_valid_year = {dataset_start}
- If coined/emerged AFTER {dataset_start}: set first_valid_year to that year
- Mark is_anachronistic = true if nonzero_years contains years BEFORE first_valid_year

### Check B — Magnitude plausibility
Examine max_value and annual_means.
- Monthly counts for a RARE mental disorder technology should typically be 0–200
- Values > 500 in early years (pre-2010) for a niche or recently coined technology \
are highly suspicious
- Mark magnitude_issue = true and set implausible_before_year if values are \
implausibly large in specific windows

### Check C — Duplication suspicion
If multiple PTs in this batch have suspiciously similar patterns (identical nonzero_years, \
very similar annual_means sequences), flag them.
- Mark duplicate_suspect = true and name the other suspected PT

## RATIONALE REQUIREMENT
For each PT, fill "reasoning" with a concise evidence-based rationale BEFORE
giving your verdict. Summarize:
1. What the web search shows about when this concept was coined or emerged
2. Whether the nonzero_years match that timeline
3. Whether the magnitudes are plausible
4. Whether anything looks copy-pasted

## CONFIDENCE RULES
- confidence = "high"   : You found a specific dated source (paper, press release, guideline)
- confidence = "medium" : You are reasonably sure but could not find a precise date
- confidence = "low"    : Genuinely uncertain; do NOT apply auto-correction for these

## INPUT DATA
{entries_json}

## OUTPUT FORMAT
Return ONLY a valid JSON array. No markdown fences, no prose outside the array. \
Each element must have EXACTLY these keys:

{{
  "name":                 string,   // PT name as given
  "column":               string,   // raw column name as given
  "reasoning":            string,   // concise evidence-based rationale (2-4 sentences)
  "search_evidence":      string,   // what your web search found (cite year + source type)
  "first_valid_year":     integer,  // year >= {dataset_start}
  "is_anachronistic":     boolean,  // true if data has nonzero values before first_valid_year
  "magnitude_issue":      boolean,  // true if max_value is implausible for the time period
  "implausible_before_year": integer | null,  // year before which values are too high; null if no issue
  "duplicate_suspect":    boolean,  // true if pattern looks copied from another PT
  "suspected_duplicate_of": string | null,   // name of the PT it may duplicate; null if none
  "confidence":           string,   // "high" | "medium" | "low"
  "correction_action":    string    // "zero_before_year" | "flag_magnitude" | "flag_duplicate" | "none"
}}

## EXAMPLES

Correct example for a real anachronism found via search:
{{
  "name": "Digital Phenotyping",
  "column": "PT_Digital Phenotyping_NoM",
  "reasoning": "Searched PubMed and found Onnela & Rauch coined the term in a 2016 JAMA Psychiatry paper. The data shows means of 3.1 in 2004 which is impossible. Magnitudes look plausible post-2016. No duplication suspected.",
  "search_evidence": "Onnela JP, Rauch SL. Harnessing smartphone-based digital phenotyping to enhance behavioral and mental health. Neuropsychopharmacology. 2016.",
  "first_valid_year": 2016,
  "is_anachronistic": true,
  "magnitude_issue": false,
  "implausible_before_year": null,
  "duplicate_suspect": false,
  "suspected_duplicate_of": null,
  "confidence": "high",
  "correction_action": "zero_before_year"
}}

Correct example for a well-established technology:
{{
  "name": "Cognitive Behavioral Therapy",
  "column": "PT_Cognitive Behavioral Therapy_NoM",
  "reasoning": "CBT was developed by Aaron Beck in the 1960s and has been mainstream since the 1980s. Present in all years makes complete sense. Magnitudes 5-50 per month are plausible for a rare-disease context.",
  "search_evidence": "Beck AT developed CBT 1960s; in widespread clinical use globally by 1980s.",
  "first_valid_year": {dataset_start},
  "is_anachronistic": false,
  "magnitude_issue": false,
  "implausible_before_year": null,
  "duplicate_suspect": false,
  "suspected_duplicate_of": null,
  "confidence": "high",
  "correction_action": "none"
}}
"""


def _build_duplicate_prompt(pairs: List[dict]) -> str:
    """
    Standalone duplicate prompt — used only when high-correlation pairs exist.
    Kept separate because duplicate pairs are detected by Python statistics first,
    then sent to Gemini for domain-knowledge verdict.
    """
    pairs_json = json.dumps(pairs, indent=2)
    return f"""You are a senior medical data quality auditor.

## TASK
The following PT (Pertinent Technology) column pairs from a mental health research \
dataset have Pearson correlation ≥ 0.9999 across 264 monthly observations — \
statistically, they are nearly identical series.

Use Google Search to investigate whether these are:
  A) "genuine_correlation"  — two conceptually distinct technologies that happen to \
trend together in research literature (rare but possible for tightly linked fields)
  B) "data_error"           — the same underlying values were copy-pasted under two \
different column names (template reuse, mislabelling, or pipeline bug)

## RATIONALE REQUIREMENT
For each pair, provide a concise evidence-based rationale that addresses:
1. Whether these two concepts are genuinely distinct in medical/technology literature
2. Whether it is scientifically plausible that they would move in near-perfect lockstep
3. Whether copy-paste/template reuse is the simpler explanation

## INPUT DATA
{pairs_json}

## OUTPUT FORMAT
Return ONLY a valid JSON array. Each element:
{{
  "col_a":      string,
  "col_b":      string,
  "reasoning":  string,   // concise rationale, 2-3 sentences
  "verdict":    string,   // "genuine_correlation" or "data_error"
  "confidence": string,   // "high" | "medium" | "low"
  "reason":     string    // one-sentence plain-English summary
}}
"""


def _parse_json_array(text: str, label: str) -> List[dict]:
    """Shared parser for any LLM response that should be a JSON array."""
    # Strip markdown fences and leading/trailing whitespace
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find all [...] blocks and try each, largest first (handles preamble/thinking text)
    candidates = list(re.finditer(r"\[.*?\]", text, re.DOTALL))
    # Also try greedy match to catch the outermost array
    greedy = re.search(r"\[.*\]", text, re.DOTALL)
    if greedy:
        candidates.append(greedy)

    # Sort by length descending — the full answer array is always the biggest block
    candidates_sorted = sorted(candidates, key=lambda m: len(m.group()), reverse=True)
    for match in candidates_sorted:
        try:
            result = json.loads(match.group())
            if isinstance(result, list) and len(result) > 0:
                return result
        except Exception:
            pass

    logger.warning(f"Could not parse {label} response as JSON array.")
    return []


def _compute_pt_stats(
    df: pd.DataFrame, date_col: str, value_col: str
) -> dict:
    """
    Compute rich per-PT statistics to embed in the master audit prompt.
    Returns annual_means, max_value, and nonzero_years so Gemini can reason
    about both the timeline and the magnitude of the data.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({"year": dates.dt.year, "val": values})
    annual = tmp.groupby("year")["val"].mean().round(3)
    annual_means = {str(int(k)): float(v) for k, v in annual.items()}
    max_value = float(values.max())
    nonzero_years = sorted(
        str(int(y)) for y, m in annual.items() if m > 0.0
    )
    return {
        "annual_means": annual_means,
        "max_value": round(max_value, 2),
        "nonzero_years": nonzero_years,
    }


def _extract_retry_seconds(exc: Exception) -> Optional[float]:
    """Parse the retry delay hint from a 429 rate-limit error message."""
    match = re.search(r"retry[_\s]delay\s*\{[^}]*seconds:\s*(\d+)", str(exc))
    if match:
        return float(match.group(1)) + 2.0  # add buffer
    return None


def _gemini_call_with_retry(
    prompt: str,
    api_key: str,
    base_retry_delay: float = 15.0,
    max_attempts: int = 3,
    use_grounding: bool = True,
) -> Optional[str]:
    """Call Gemini with exponential backoff. Returns raw text or None on failure."""
    for attempt in range(max_attempts):
        try:
            return _gemini_chat(prompt, api_key, use_grounding=use_grounding)
        except Exception as exc:
            if _is_gemini_auth_error(exc):
                raise GeminiAuthenticationError(
                    "Gemini API key is invalid or expired. Renew GEMINI_API_KEY "
                    "or run with --skip_validation/--no_gemini for exploratory output."
                ) from exc
            logger.warning(f"Gemini API attempt {attempt + 1} failed: {exc}")
            if attempt < max_attempts - 1:
                wait = _extract_retry_seconds(exc) or base_retry_delay * (attempt + 1)
                logger.info(f"  Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
    return None


def _query_gemini_master_audit(
    df: pd.DataFrame,
    date_col: str,
    pt_cols: List[str],
    col_to_clean: Dict[str, str],
    api_key: str,
    batch_size: int = 10,
    base_retry_delay: float = 15.0,
    use_grounding: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Master audit query — sends the rich master prompt for every uncovered PT.

    Each batch of `batch_size` PTs is sent in one call with:
      - PT name + raw column name
      - annual mean values (for timeline reasoning)
      - monthly peak value (for magnitude plausibility check)
      - list of nonzero years (for quick anachronism spotting)

    Gemini is asked to perform all three checks (anachronism, magnitude, duplicate)
    and return concise rationale + confidence + correction_action per PT.

    Only items with confidence != "low" are actioned by the caller.
    """
    result: Dict[str, dict] = {}
    col_list = list(pt_cols)

    for i in range(0, len(col_list), batch_size):
        batch_cols = col_list[i : i + batch_size]
        entries = []
        for col in batch_cols:
            stats = _compute_pt_stats(df, date_col, col)
            entries.append({
                "name": col_to_clean[col],
                "column": col,
                **stats,
            })

        prompt = _build_master_audit_prompt(entries)
        raw = _gemini_call_with_retry(
            prompt, api_key,
            base_retry_delay=base_retry_delay,
            use_grounding=use_grounding,
        )
        if raw is None:
            continue

        parsed = _parse_json_array(raw, "master_audit")
        for item in parsed:
            # Accept match by column name (most reliable) or clean name
            col = item.get("column", "")
            if col not in {c: None for c in batch_cols}:
                # Try matching by clean name as fallback
                clean_to_col = {v: k for k, v in col_to_clean.items() if k in batch_cols}
                col = clean_to_col.get(item.get("name", ""), "")
            if col:
                result[col] = item

    return result


def _query_gemini_duplicates(
    df: pd.DataFrame,
    date_col: str,
    duplicate_pairs: List[Tuple[str, str, float]],
    api_key: str,
    batch_size: int = 8,
    base_retry_delay: float = 15.0,
    use_grounding: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """
    Standalone duplicate verdict query — only called when Python statistics have
    already identified high-correlation pairs (corr >= _DUP_CORR_THRESHOLD).
    Sends the standalone duplicate prompt with concise rationale + grounding.
    """
    results: List[dict] = []

    for i in range(0, len(duplicate_pairs), batch_size):
        batch = duplicate_pairs[i : i + batch_size]
        pairs_payload = []
        for col_a, col_b, corr in batch:
            stats_a = _compute_pt_stats(df, date_col, col_a)
            stats_b = _compute_pt_stats(df, date_col, col_b)
            pairs_payload.append({
                "col_a": col_a,
                "col_b": col_b,
                "correlation": round(corr, 6),
                "annual_means_a": stats_a["annual_means"],
                "annual_means_b": stats_b["annual_means"],
            })

        prompt = _build_duplicate_prompt(pairs_payload)
        raw = _gemini_call_with_retry(
            prompt, api_key,
            base_retry_delay=base_retry_delay,
            use_grounding=use_grounding,
        )
        if raw is None:
            continue

        parsed = _parse_json_array(raw, "duplicate")
        results.extend(parsed)

    return results


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _find_duplicate_columns(
    df: pd.DataFrame,
    pt_cols: List[str],
) -> List[Tuple[str, str, float]]:
    """
    Return pairs of PT columns whose numeric values are nearly identical
    (Pearson correlation >= _DUP_CORR_THRESHOLD).
    """
    duplicates: List[Tuple[str, str, float]] = []
    data = df[pt_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat = data.values  # shape [T, N]
    n = len(pt_cols)
    for i in range(n):
        for j in range(i + 1, n):
            col_i, col_j = mat[:, i], mat[:, j]
            # Skip zero-only columns
            if col_i.std() < 1e-9 or col_j.std() < 1e-9:
                continue
            corr = float(np.corrcoef(col_i, col_j)[0, 1])
            if corr >= _DUP_CORR_THRESHOLD:
                duplicates.append((pt_cols[i], pt_cols[j], corr))
    return duplicates


# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def _clean_pt_name(col: str) -> str:
    """Convert raw CSV column name to readable PT name."""
    s = re.sub(r"^PT[_\-\s]+", "", col, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoM$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoP$", "", s, flags=re.IGNORECASE)
    return s.replace("_", " ").strip().title()


def _col_to_clean(cols: List[str]) -> Dict[str, str]:
    return {c: _clean_pt_name(c) for c in cols}



def _clean_feature_name(col: str) -> str:
    """Convert any RMD/PT CSV column name to a normalized display name."""
    s = re.sub(r"^(RMD|PT)[_\-\s]+", "", str(col), flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoM$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_\-\s]+NoP$", "", s, flags=re.IGNORECASE)
    return s.replace("_", " ").strip().title()


def _clinical_columns_by_name(df: pd.DataFrame, prefix: str) -> Dict[str, List[str]]:
    pattern = rf"^{prefix}[_\-\s]"
    result: Dict[str, List[str]] = {}
    for col in df.columns:
        if re.match(pattern, str(col), re.IGNORECASE):
            result.setdefault(_clean_feature_name(str(col)), []).append(str(col))
    return result


def apply_hard_constraints(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Zero-out known invalid PT signals when their blacklisted RMD context exists.

    The BR-MTGNN CSV is not a pairwise RMD-by-PT matrix. RMDs and PTs are
    separate time-series columns, so when a blacklisted RMD column and PT column
    are both present, the enforceable hard constraint at this stage is to zero
    the matching PT column globally before Gemini validation.
    """
    if verbose:
        print("[gemini_validator] Applying hard-constraint clinical blacklist...")

    rmd_cols_by_name = _clinical_columns_by_name(df, "RMD")
    pt_cols_by_name = _clinical_columns_by_name(df, "PT")
    report: List[dict] = []

    for pt, reason in CLINICAL_GLOBAL_PT_BLACKLIST.items():
        pt_key = pt.strip().title()
        for pt_col in pt_cols_by_name.get(pt_key, []):
            values = pd.to_numeric(df[pt_col], errors="coerce").fillna(0.0)
            rows_zeroed = int((values != 0.0).sum())
            df.loc[:, pt_col] = 0.0
            report.append({
                "source": "clinical_global_blacklist",
                "check": "hard_constraint",
                "column": pt_col,
                "pt_name": pt,
                "rows_zeroed": rows_zeroed,
                "action": "ZERO",
                "reason": reason,
            })
            if verbose:
                print(f"[gemini_validator] GLOBAL BLACKLIST ZERO: {pt_col} ({rows_zeroed} nonzero rows).")

    for rmd, forbidden_pts in CLINICAL_BLACKLIST.items():
        rmd_key = rmd.strip().title()
        if rmd_key not in rmd_cols_by_name:
            continue
        for pt in forbidden_pts:
            pt_key = pt.strip().title()
            for pt_col in pt_cols_by_name.get(pt_key, []):
                values = pd.to_numeric(df[pt_col], errors="coerce").fillna(0.0)
                rows_zeroed = int((values != 0.0).sum())
                df.loc[:, pt_col] = 0.0
                item = {
                    "source": "clinical_blacklist",
                    "check": "hard_constraint",
                    "rmd_name": rmd,
                    "rmd_columns": rmd_cols_by_name[rmd_key],
                    "column": pt_col,
                    "pt_name": pt,
                    "rows_zeroed": rows_zeroed,
                    "action": "ZERO",
                    "reason": f"Hard clinical blacklist: {pt} is not a plausible pairing for {rmd}.",
                }
                report.append(item)
                if verbose:
                    print(f"[gemini_validator] BLACKLIST ZERO: {pt_col} for {rmd} ({rows_zeroed} nonzero rows).")

    df.attrs["_clinical_blacklist_report"] = report
    return df


# ---------------------------------------------------------------------------
# Core correction logic
# ---------------------------------------------------------------------------

def _zero_before_year(
    df: pd.DataFrame,
    date_col: Optional[str],
    col: str,
    first_valid_year: int,
) -> int:
    """
    Zero-out values in `col` for all rows whose date is before `first_valid_year`.
    Returns the number of cells zeroed.
    """
    if date_col is None:
        return 0

    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    mask = dates.dt.year < first_valid_year
    zeroed = int(mask.sum())
    if zeroed > 0:
        df.loc[mask, col] = 0.0
        logger.info(f"  Zeroed {zeroed} rows in '{col}' (before {first_valid_year})")
    return zeroed


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    date_keywords = {"date", "month", "time", "timestamp", "ds",
                     "month-year", "month_year", "year-month"}
    for c in df.columns:
        if str(c).strip().lower() in date_keywords:
            return c
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_correct(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    use_gemini: bool = True,
    verbose: bool = True,
    cache_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Validate historical PT data and zero-out anachronistic values.

    Parameters
    ----------
    df         : Raw data DataFrame (data/data.csv contents).
    api_key    : Gemini API key. Falls back to GEMINI_API_KEY env var.
    use_gemini : If False, skip all anachronism corrections.
                 Duplicate detection still runs without the API.
    verbose    : Print progress to stdout.
    cache_path : Path to the JSON cache file. Defaults to data/validation_cache.json.
                 Keyed by SHA-256 of PT column values — auto-invalidates on data change.

    Returns
    -------
    corrected_df : Copy of df with anachronistic values zeroed.
    report       : List of dicts describing each correction applied.
    """
    df = df.copy()
    df = apply_hard_constraints(df, verbose=verbose)
    report: List[dict] = list(df.attrs.pop("_clinical_blacklist_report", []))
    date_col = _detect_date_col(df)

    if verbose:
        print("[gemini_validator] Starting historical trend audit...")
        if not use_gemini:
            print("[gemini_validator] NOTE: Gemini disabled (--no_gemini). "
                  "Only hard constraints and duplicate detection will run.")
        if date_col:
            print(f"[gemini_validator] Date column: '{date_col}'")
        else:
            print("[gemini_validator] WARNING: No date column found; anachronism corrections skipped.")

    # Identify PT columns
    pt_cols = [c for c in df.columns if re.match(r"^PT[_\-\s]", str(c), re.IGNORECASE)]
    if verbose:
        print(f"[gemini_validator] Found {len(pt_cols)} PT columns.")

    col_to_clean = _col_to_clean(pt_cols)

    # ------------------------------------------------------------------
    # Gemini master audit (Google Search grounded)
    # Covers: anachronism, magnitude plausibility, duplicate suspicion.
    # Results cached by SHA-256 of PT data — re-runs skip the API entirely.
    # ------------------------------------------------------------------
    gemini_corrected: Dict[str, int] = {}
    if use_gemini:
        _load_dotenv()
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            if verbose:
                print("[gemini_validator] WARNING: No GEMINI_API_KEY found. "
                      "Skipping audit.")
        elif date_col is None:
            if verbose:
                print("[gemini_validator] WARNING: No date column — skipping audit.")
        else:
            _cache_path = cache_path if cache_path is not None else _DEFAULT_CACHE_PATH
            data_hash = _compute_data_hash(df, pt_cols)
            cached = _load_cache(_cache_path, data_hash)

            if cached is not None:
                if verbose:
                    print(f"[gemini_validator] Cache hit ({_cache_path.name}) — "
                          f"skipping Gemini API call. Delete '{_cache_path}' to force re-audit.")
                audit_results = cached
            else:
                if verbose:
                    print(f"[gemini_validator] Gemini audit (Google Search grounded): "
                          f"querying {len(pt_cols)} PTs in batches of 10...")
                try:
                    audit_results = _query_gemini_master_audit(
                        df, date_col, pt_cols, col_to_clean, key, verbose=verbose,
                    )
                except GeminiAuthenticationError as exc:
                    if verbose:
                        print(f"[gemini_validator] WARNING: {exc}")
                    audit_results = {}
                except Exception as exc:
                    logger.warning(f"Gemini audit failed: {exc}")
                    audit_results = {}

                if audit_results:
                    _save_cache(_cache_path, data_hash, audit_results)
                    if verbose:
                        print(f"[gemini_validator] Audit results cached to '{_cache_path}'.")

            dataset_start = 2004
            for col, item in audit_results.items():
                confidence = item.get("confidence", "low")
                action = item.get("correction_action", "none")
                reasoning = item.get("reasoning", "")
                evidence = item.get("search_evidence", "")
                pt_name = col_to_clean.get(col, col)

                if action == "zero_before_year" and confidence in ("high", "medium"):
                    year = item.get("first_valid_year", dataset_start)
                    try:
                        year = int(year)
                    except (TypeError, ValueError):
                        continue
                    if year <= dataset_start:
                        continue
                    gemini_corrected[col] = year
                    n = _zero_before_year(df, date_col, col, year)
                    report.append({
                        "source": "gemini_master_audit",
                        "check": "anachronism",
                        "column": col,
                        "pt_name": pt_name,
                        "first_valid_year": year,
                        "confidence": confidence,
                        "gemini_reasoning": reasoning,
                        "search_evidence": evidence,
                        "rows_zeroed": n,
                    })

                if item.get("magnitude_issue", False):
                    impl_year = item.get("implausible_before_year")
                    report.append({
                        "source": "gemini_master_audit",
                        "check": "magnitude",
                        "column": col,
                        "pt_name": pt_name,
                        "implausible_before_year": impl_year,
                        "confidence": confidence,
                        "gemini_reasoning": reasoning,
                        "search_evidence": evidence,
                        "action": "flag_magnitude",
                    })
                    if verbose:
                        print(f"[gemini_validator] MAGNITUDE ISSUE [{confidence}]: "
                              f"'{col}' — values implausible before {impl_year}. "
                              f"{reasoning}")

                if item.get("duplicate_suspect", False):
                    suspected = item.get("suspected_duplicate_of", "unknown")
                    report.append({
                        "source": "gemini_master_audit",
                        "check": "duplicate_suspect",
                        "column": col,
                        "pt_name": pt_name,
                        "suspected_duplicate_of": suspected,
                        "confidence": confidence,
                        "gemini_reasoning": reasoning,
                        "action": "flagged_for_review",
                    })
                    if verbose:
                        print(f"[gemini_validator] DUPLICATE SUSPECT [{confidence}]: "
                              f"'{col}' may duplicate '{suspected}'. {reasoning}")

                if confidence == "low" and action != "none":
                    report.append({
                        "source": "gemini_master_audit",
                        "check": "low_confidence_flag",
                        "column": col,
                        "pt_name": pt_name,
                        "intended_action": action,
                        "gemini_reasoning": reasoning,
                        "action": "flagged_for_human_review",
                    })

            anachronism_count = sum(
                1 for r in report
                if r.get("source") == "gemini_master_audit" and r.get("check") == "anachronism"
            )
            if verbose:
                print(f"[gemini_validator] Gemini audit complete: "
                      f"{anachronism_count} anachronism corrections, "
                      f"{len(audit_results) - anachronism_count} flags.")

    # ------------------------------------------------------------------
    # Step 3: Duplicate / cross-contamination detection (pure Python first)
    # Then Prompt 2 — send high-corr pairs to Gemini to distinguish
    # genuine co-movement from data copy-paste errors.
    # ------------------------------------------------------------------
    dups = _find_duplicate_columns(df, pt_cols)

    if dups and use_gemini and date_col:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if key:
            if verbose:
                print(f"[gemini_validator] Duplicate verdict: "
                      f"sending {len(dups)} high-correlation pair(s) to Gemini...")
            try:
                dup_verdicts = _query_gemini_duplicates(
                    df, date_col, dups, key, verbose=verbose,
                )
            except Exception as exc:
                logger.warning(f"Gemini duplicate query failed: {exc}")
                dup_verdicts = []

            for verdict in dup_verdicts:
                col_a = verdict.get("col_a", "")
                col_b = verdict.get("col_b", "")
                v = verdict.get("verdict", "")
                confidence = verdict.get("confidence", "")
                reason = verdict.get("reason", "")
                action = "flagged_for_review"
                if v == "data_error" and confidence == "high":
                    action = "data_error_confirmed"
                if verbose:
                    tag = "DATA ERROR" if v == "data_error" else "genuine correlation"
                    print(f"[gemini_validator] DUPLICATE [{tag}, {confidence}]: "
                          f"'{col_a}' vs '{col_b}' — {reason}")
                report.append({
                    "source": "gemini_duplicate_check",
                    "column_a": col_a,
                    "column_b": col_b,
                    "gemini_verdict": v,
                    "confidence": confidence,
                    "gemini_reason": reason,
                    "action": action,
                })
        else:
            for col_a, col_b, corr in dups:
                if verbose:
                    print(f"[gemini_validator] DUPLICATE (corr={corr:.6f}): "
                          f"'{col_a}' vs '{col_b}' — flagged for review (no Gemini key)")
                report.append({
                    "source": "duplicate_detection",
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": corr,
                    "action": "flagged_for_review",
                })
    else:
        for col_a, col_b, corr in dups:
            if verbose:
                print(f"[gemini_validator] DUPLICATE (corr={corr:.6f}): "
                      f"'{col_a}' vs '{col_b}' — flagged for review")
            report.append({
                "source": "duplicate_detection",
                "column_a": col_a,
                "column_b": col_b,
                "correlation": corr,
                "action": "flagged_for_review",
            })

    # ------------------------------------------------------------------
    # Step 4: Self-referential PT columns — PT_<RMD_Name>
    # A PT column whose name matches an RMD (e.g. PT_Chronic_Traumatic_
    # Encephalopathy_NoM alongside RMD_Chronic_Traumatic_Encephalopathy_NoM)
    # causes a duplicate legend entry and a meaningless self-referential trend.
    # These columns are DROPPED from the dataframe, not merely flagged.
    # ------------------------------------------------------------------
    rmd_names = {
        re.sub(r"^RMD[_\-\s]+", "", re.sub(r"[_\-\s]+NoM$", "", c, flags=re.IGNORECASE), flags=re.IGNORECASE).lower()
        for c in df.columns if re.match(r"^RMD[_\-\s]", str(c), re.IGNORECASE)
    }
    cols_to_drop: List[str] = []
    for col in pt_cols:
        clean = _clean_pt_name(col).lower()
        if clean in rmd_names:
            cols_to_drop.append(col)
            if verbose:
                print(
                    f"[gemini_validator] SELF-REFERENTIAL PT REMOVED: '{col}' shares "
                    f"its name with an RMD — column dropped to prevent duplicate legend entries."
                )
            report.append({
                "source": "label_duplicate",
                "column": col,
                "pt_name": _clean_pt_name(col),
                "action": "column_dropped",
            })

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        if verbose:
            print(f"[gemini_validator] Dropped {len(cols_to_drop)} self-referential PT column(s).")

    if verbose:
        total_corrections = sum(1 for r in report if r.get("rows_zeroed", 0) > 0)
        print(
            f"[gemini_validator] Audit complete. "
            f"{total_corrections} columns corrected, "
            f"{len(dups)} duplicates flagged."
        )

    return df, report


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Validate and correct BR-MTGNN historical PT data via Gemini.")
    p.add_argument("--input_csv", type=str, default="data/data.csv")
    p.add_argument("--output_csv", type=str, default="data/data_validated.csv")
    p.add_argument("--gemini_api_key", type=str, default="",
                   help="Gemini API key (overrides GEMINI_API_KEY env var)")
    p.add_argument("--no_gemini", action="store_true",
                   help="Skip Gemini API call; no anachronism corrections applied (duplicate detection still runs)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print audit report only; do not write output CSV")
    p.add_argument("--report_json", type=str, default="",
                   help="Optional path to write the JSON audit report")
    p.add_argument("--cache_path", type=str, default="",
                   help="Path to validation cache JSON (default: data/validation_cache.json)")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable cache; always call the Gemini API")
    return p.parse_args()


def main():
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"[gemini_validator] Loaded {args.input_csv}: {df.shape[0]} rows × {df.shape[1]} cols")

    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    cache_path = None if args.no_cache else (Path(args.cache_path) if args.cache_path else None)
    corrected_df, report = validate_and_correct(
        df,
        api_key=api_key or None,
        use_gemini=not args.no_gemini,
        verbose=True,
        cache_path=cache_path,
    )

    if args.report_json:
        from pathlib import Path
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"[gemini_validator] Report written to {args.report_json}")

    if args.dry_run:
        print("[gemini_validator] Dry run — no output file written.")
        sys.exit(0)

    from pathlib import Path
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    corrected_df.to_csv(args.output_csv, index=False)
    print(f"[gemini_validator] Corrected data saved to {args.output_csv}")


if __name__ == "__main__":
    main()
