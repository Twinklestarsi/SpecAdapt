#!/usr/bin/env python3
"""Run the 10 test specs through the analyzer and collect results."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spec_analyze import Analyzer

TEST_SPECS = Path(__file__).resolve().parent / "test_specs.json"
OUTPUT = Path(__file__).resolve().parent / "test_specs_results.json"


def main():
    with open(TEST_SPECS) as f:
        specs = json.load(f)

    analyzer = Analyzer()
    print(f"Endpoint: {analyzer.endpoint}")
    print(f"Model:    {analyzer.model}")
    print(f"Specs:    {len(specs)}\n")

    results = []
    for i, spec_entry in enumerate(specs):
        sid = spec_entry["id"]
        detail = spec_entry["detail_level"]
        text = spec_entry["spec"]

        print(f"[{i+1}/{len(specs)}] {sid} (detail: {detail})")
        print(f"  Input: {text[:80]}{'...' if len(text) > 80 else ''}")

        r = analyzer.analyze_spec(text)

        feat = r.llm_features
        if feat:
            print(f"  -> pattern={feat.get('architecture_pattern')} "
                  f"complexity={feat.get('complexity')} "
                  f"sequential={feat.get('is_sequential')} "
                  f"fsm={feat.get('has_fsm')}({feat.get('estimated_fsm_states')})")
            print(f"     subcategory={feat.get('suggested_subcategory')} "
                  f"width={feat.get('data_width_dominant')} "
                  f"clocks={feat.get('num_clock_domains')}")
            print(f"     purpose: {feat.get('purpose', '')[:70]}")
        else:
            print(f"  -> LLM ERROR")

        results.append({
            "id": sid,
            "detail_level": detail,
            "spec_text": text,
            "spec_word_count": len(text.split()),
            "llm_features": r.llm_features,
            "confidence": r.confidence,
            "overall_confidence": r.overall_confidence,
            "token_usage": r.token_usage,
        })

        if i < len(specs) - 1:
            time.sleep(1.5)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(results)} results to {OUTPUT}")


if __name__ == "__main__":
    main()
