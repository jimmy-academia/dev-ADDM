#!/usr/bin/env python3
"""
AMOS Evidence Filter Fix - Parallel Script

This script fixes the AMOS evidence generation bug where ALL extracted fields
were being reported as evidence, even for reviews with no actual incidents.

The fix: Only create evidence entries for reviews where incident_severity != "none".

Usage:
    .venv/bin/python scripts/debug/fix_amos_evidence.py [--apply] [--test]

Options:
    --apply   Apply the fix to phase2.py (creates backup first)
    --test    Run a quick AMOS test after applying
    --dry-run Show what would change without modifying files (default)
"""

import argparse
import shutil
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE2_PATH = PROJECT_ROOT / "src" / "addm" / "methods" / "amos" / "phase2.py"

# The problematic code (to find)
OLD_CODE = '''        for ext in self._extractions:
            review_id = ext.get("review_id", ext.get("_review_id", "unknown"))
            snippet = ext.get("_snippet", "")

            # If no snippet stored, try supporting_quote field or extraction metadata
            if not snippet:
                snippet = ext.get("supporting_quote", "")
            if not snippet and "_source_text" in ext:
                snippet = ext["_source_text"][:200]  # Limit snippet length

            for field, value in ext.items():
                # Skip internal fields, metadata, and supporting_quote (stored as _snippet)
                if field.startswith("_") or field in ("review_id", "is_relevant", "supporting_quote"):
                    continue

                evidences.append({
                    "evidence_id": f"E{evidence_idx}",
                    "review_id": review_id,
                    "field": field.lower(),
                    "judgement": str(value).lower() if value else "none",
                    "snippet": snippet,
                })
                evidence_idx += 1'''

# The fixed code (to replace with)
NEW_CODE = '''        for ext in self._extractions:
            review_id = ext.get("review_id", ext.get("_review_id", "unknown"))
            snippet = ext.get("_snippet", "")

            # If no snippet stored, try supporting_quote field or extraction metadata
            if not snippet:
                snippet = ext.get("supporting_quote", "")
            if not snippet and "_source_text" in ext:
                snippet = ext["_source_text"][:200]  # Limit snippet length

            # Check if this extraction represents an actual incident
            # Only create evidence for reviews with incident_severity != "none"
            severity = ext.get("incident_severity", ext.get("severity", "none"))
            severity_normalized = str(severity).strip().lower() if severity else "none"

            # Normalize common "no incident" variations to "none"
            if severity_normalized in _SEVERITY_NORMALIZATION_MAP:
                severity_normalized = _SEVERITY_NORMALIZATION_MAP[severity_normalized]
            elif any(ind in severity_normalized for ind in ["no ", "none", "n/a", "not ", "absent", "nothing"]):
                severity_normalized = "none"

            # Skip extractions that don't represent actual incidents
            if severity_normalized == "none":
                continue

            for field, value in ext.items():
                # Skip internal fields, metadata, and supporting_quote (stored as _snippet)
                if field.startswith("_") or field in ("review_id", "is_relevant", "supporting_quote"):
                    continue

                evidences.append({
                    "evidence_id": f"E{evidence_idx}",
                    "review_id": review_id,
                    "field": field.lower(),
                    "judgement": str(value).lower() if value else "none",
                    "snippet": snippet,
                })
                evidence_idx += 1'''


def check_fix_needed() -> bool:
    """Check if the fix has already been applied."""
    content = PHASE2_PATH.read_text()
    # If the old code is present, fix is needed
    if OLD_CODE in content:
        return True
    # If the new code is present, already fixed
    if "# Skip extractions that don't represent actual incidents" in content:
        return False
    # Unknown state
    print("WARNING: Could not determine current state of phase2.py")
    return False


def show_diff():
    """Show what the fix changes."""
    print("=" * 70)
    print("AMOS EVIDENCE FILTER FIX")
    print("=" * 70)
    print()
    print("ROOT CAUSE:")
    print("  _build_standard_output() creates evidence for ALL fields in EVERY")
    print("  extraction, even when incident_severity='none' (no actual incident).")
    print()
    print("  Evaluation counts all evidences as 'claimed incidents' and compares")
    print("  against GT incidents, causing 0% precision when claiming non-incidents.")
    print()
    print("THE FIX:")
    print("  Add a check to skip extractions where incident_severity == 'none'")
    print()
    print("LOCATION: src/addm/methods/amos/phase2.py in _build_standard_output()")
    print()
    print("-" * 70)
    print("BEFORE (lines ~1329-1351):")
    print("-" * 70)
    for i, line in enumerate(OLD_CODE.split('\n')[:15], 1):
        print(f"  {line}")
    print("  ...")
    print()
    print("-" * 70)
    print("AFTER (added severity check before field loop):")
    print("-" * 70)
    for i, line in enumerate(NEW_CODE.split('\n')[:25], 1):
        print(f"  {line}")
    print("  ...")
    print()


def apply_fix(dry_run: bool = True) -> bool:
    """Apply the fix to phase2.py."""
    if not check_fix_needed():
        print("Fix already applied or code changed. No action needed.")
        return True

    content = PHASE2_PATH.read_text()

    if OLD_CODE not in content:
        print("ERROR: Could not find the target code block to replace.")
        print("The code may have been modified. Manual fix required.")
        return False

    if dry_run:
        print("DRY RUN: Would replace code block in phase2.py")
        show_diff()
        return True

    # Create backup
    backup_path = PHASE2_PATH.with_suffix('.py.bak')
    shutil.copy(PHASE2_PATH, backup_path)
    print(f"Created backup: {backup_path}")

    # Apply fix
    new_content = content.replace(OLD_CODE, NEW_CODE)
    PHASE2_PATH.write_text(new_content)
    print(f"Applied fix to: {PHASE2_PATH}")

    return True


def run_test():
    """Run a quick AMOS test to verify the fix."""
    import subprocess

    print()
    print("=" * 70)
    print("RUNNING AMOS TEST")
    print("=" * 70)
    print()

    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-m", "addm.tasks.cli.run_experiment",
        "--policy", "G1_allergy_V2",
        "-n", "3",
        "--method", "amos",
        "--dev"
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Fix AMOS evidence generation bug")
    parser.add_argument("--apply", action="store_true", help="Apply the fix (creates backup)")
    parser.add_argument("--test", action="store_true", help="Run AMOS test after fix")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show changes without applying (default)")
    args = parser.parse_args()

    # If --apply is specified, disable dry-run
    dry_run = not args.apply

    print(f"Phase2 path: {PHASE2_PATH}")
    print(f"Fix needed: {check_fix_needed()}")
    print()

    if dry_run and not args.test:
        show_diff()
        print()
        print("Run with --apply to apply the fix.")
        print("Run with --test to run AMOS test (without applying fix).")
        print("Run with --apply --test to apply and test.")
        return 0

    if args.apply:
        if not apply_fix(dry_run=False):
            return 1

    if args.test:
        if not run_test():
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
