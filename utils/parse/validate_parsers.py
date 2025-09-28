#!/usr/bin/env python3

import importlib
import os
from pathlib import Path


def analyze_rules(rules_list, parser_module, name):
    """Analyze rules by length and show extremes"""
    print(f"\n=== {name} Analysis ===")

    # Use the parser's flatten function
    flattened = parser_module.flatten(rules_list)

    # Filter out rules without IDs and get title lengths
    rules_with_lengths = []
    for rule in flattened:
        if rule["id"] and rule["title"]:
            title_len = len(rule["title"])
            rules_with_lengths.append((rule["id"], rule["title"], title_len))

    # Sort by length
    rules_by_length = sorted(rules_with_lengths, key=lambda x: x[2])

    print(f"Total rules with IDs: {len(rules_with_lengths)}")
    print(
        f"Average title length: {sum(x[2] for x in rules_with_lengths) / len(rules_with_lengths):.1f}"
    )

    print(f"\n🔍 TOP 5 SHORTEST RULES:")
    for i, (rule_id, title, length) in enumerate(rules_by_length[:5]):
        print(f"{i + 1}. [{rule_id}] ({length} chars): {title}")

    print(f"\n🔍 TOP 5 LONGEST RULES:")
    for i, (rule_id, title, length) in enumerate(rules_by_length[-5:]):
        print(f"{i + 1}. [{rule_id}] ({length} chars): {title}")
        if length > 500:  # Show more detail for very long rules
            print(f"   Preview: {title[:200]}...")


def discover_parsers():
    """Automatically discover all parse_* modules and their corresponding rules files"""
    parsers = {}

    # Find all parse_*.py files
    for file in Path(".").glob("parse_*.py"):
        if file.name == "parse_ultimate.py":
            continue  # Skip base file, handle separately

        module_name = file.stem  # e.g., "parse_discgolf"
        sport_name = (
            module_name.replace("parse_", "").replace("_", " ").title()
        )  # e.g., "Disc Golf"

        # Special case formatting for known sports
        if sport_name == "Discgolf":
            sport_name = "Disc Golf"
        rules_file = f"../rules/{module_name.replace('parse_', '')}.html"  # e.g., "../rules/discgolf.html"

        # Check if rules file exists
        if Path(rules_file).exists():
            try:
                parser_module = importlib.import_module(module_name)
                parsers[sport_name] = {
                    "module": parser_module,
                    "rules_file": rules_file,
                }
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")

    # Handle Ultimate separately (special case)
    if Path("parse_ultimate.py").exists() and Path("../rules/ultimate.html").exists():
        try:
            ultimate_module = importlib.import_module("parse_ultimate")
            parsers["Ultimate"] = {
                "module": ultimate_module,
                "rules_file": "../rules/ultimate.html",
            }
        except ImportError as e:
            print(f"Warning: Could not import parse_ultimate: {e}")

    return parsers


def cross_validate(all_results):
    """Perform cross-validation checks across all parsers"""
    print(f"\n=== Cross-Validation Checks ===")

    for name, data in all_results.items():
        flattened = data["flattened"]

        # Check for rules with empty titles
        empty = [r for r in flattened if r["id"] and not r["title"].strip()]
        print(f"{name} rules with empty titles: {len(empty)}")

        # Check for extremely short titles (likely parsing issues)
        tiny = [r for r in flattened if r["id"] and r["title"] and len(r["title"]) < 10]
        print(f"{name} rules with titles < 10 chars: {len(tiny)}")
        for rule in tiny:
            print(f"  - [{rule['id']}]: '{rule['title']}'")


def main():
    """Main validation function that auto-discovers parsers"""
    print("🔍 Auto-discovering rule parsers...")
    parsers = discover_parsers()

    if not parsers:
        print(
            "❌ No rule parsers found! Make sure parse_*.py files exist with corresponding rules/*.html files."
        )
        return

    print(f"✅ Found {len(parsers)} parser(s): {', '.join(parsers.keys())}")

    all_results = {}

    # Analyze each parser
    for sport_name, parser_info in parsers.items():
        print(f"\n📖 Loading and analyzing {sport_name} rules...")
        try:
            rules = parser_info["module"].load_rules(
                html_path=parser_info["rules_file"]
            )
            analyze_rules(rules, parser_info["module"], sport_name)
            all_results[sport_name] = {
                "rules": rules,
                "flattened": parser_info["module"].flatten(rules),
                "module": parser_info["module"],
            }
            print("✅ Success")
        except Exception as e:
            print(f"❌ Failed to analyze {sport_name}: {e}")

        print("\n" + "=" * 80)

    # Cross-validation
    if all_results:
        cross_validate(all_results)

    print(f"\n🎉 Validation complete! Analyzed {len(all_results)} rule sets.")


if __name__ == "__main__":
    main()
