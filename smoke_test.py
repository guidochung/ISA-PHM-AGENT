"""Smoke test for the upgraded prototype.

Runs every metadata-only tool against the Milling JSON and prints a one-line
status per tool. Exits non-zero if any tool crashes.
"""

import json
import sys
import traceback

from isa_parser import load_from_file
from tools import build_tool_definitions, execute_tool, FIGURE_REGISTRY


JSON_PATH = "Multi Run Milling ISA-PHM (1).json"
DATASET_NAME = "Milling"


def main() -> int:
    print(f"Loading {JSON_PATH}...")
    parser = load_from_file(JSON_PATH)
    parsers = {DATASET_NAME: parser}
    data_dirs = {DATASET_NAME: ""}  # No CSVs available - tests data_not_available paths

    studies = parser.list_studies()
    if not studies:
        print("FAIL: no studies in dataset")
        return 1
    first_study_title = studies[0]["title"]
    first_sensor = (studies[0]["sensor_aliases"] or [""])[0]
    print(f"  First study: {first_study_title}")
    print(f"  First sensor: {first_sensor}")
    print()

    tests = [
        # tool_name, input_dict, brief
        ("get_investigation_overview", {"dataset_name": DATASET_NAME}, "investigation overview"),
        ("list_studies", {"dataset_name": DATASET_NAME}, "list studies"),
        ("ai_context", {"dataset_name": DATASET_NAME}, "ai_context (semantic manifest + PHM objective)"),
        ("validate_dataset", {"dataset_name": DATASET_NAME}, "validate (gates 1, 3, 4 - file linkage skipped)"),
        ("get_experiment_matrix", {"dataset_name": DATASET_NAME}, "experiment matrix"),
        ("get_run_inventory", {"dataset_name": DATASET_NAME, "study_title": first_study_title}, f"run inventory for {first_study_title!r}"),
        ("get_label_coverage", {"dataset_name": DATASET_NAME}, "label coverage"),
        ("get_sensor_availability", {"dataset_name": DATASET_NAME}, "sensor availability"),
        ("classify_phm_objective", {"dataset_name": DATASET_NAME}, "classify PHM objective"),
        ("get_protocol_details", {"dataset_name": DATASET_NAME, "study_title": first_study_title}, "protocol details (J-series)"),
        ("get_replication_gaps", {"dataset_name": DATASET_NAME, "study_title": first_study_title}, "replication gaps (J-series)"),
        ("get_sensor_compatibility", {"dataset_name": DATASET_NAME, "study_title": first_study_title, "sensor_alias": first_sensor}, f"sensor compatibility for {first_sensor!r}"),
        ("get_study_details", {"dataset_name": DATASET_NAME, "study_title": first_study_title}, "study details"),
        ("get_assay_details", {"dataset_name": DATASET_NAME, "study_title": first_study_title, "sensor_alias": first_sensor}, "assay details"),
        ("list_data_files", {"dataset_name": DATASET_NAME, "study_title": first_study_title, "sensor_alias": first_sensor}, "list data files"),
        ("search_metadata", {"dataset_name": DATASET_NAME, "term": "wear"}, "search metadata for 'wear'"),
        ("compare_studies", {"dataset_name": DATASET_NAME, "study_titles": [s["title"] for s in studies[:2]]}, "compare first 2 studies"),
        ("load_run_csv", {"dataset_name": DATASET_NAME, "relative_path": "./Case_01/Sensor/vib_table/Case_01_time_run_01.csv"}, "load_run_csv (expect data_not_available)"),
        ("lifecycle_features", {"dataset_name": DATASET_NAME, "study_title": first_study_title, "sensor_alias": first_sensor}, "lifecycle_features (expect data_not_available)"),
        ("make_plot", {"dataset_name": DATASET_NAME, "kind": "lifecycle", "study_title": first_study_title, "sensor_alias": first_sensor}, "make_plot lifecycle (expect data_not_available)"),
    ]

    failures = []
    for tool_name, tool_input, brief in tests:
        try:
            result_str = execute_tool(tool_name, tool_input, parsers, data_dirs)
            result = json.loads(result_str)
            if isinstance(result, dict) and result.get("error"):
                print(f"  [FAIL] {tool_name}: {result['error']}")
                failures.append((tool_name, result["error"]))
            else:
                preview = ""
                if isinstance(result, dict):
                    if "status" in result:
                        preview = f" status={result['status']}"
                    elif "phm_objective" in result and isinstance(result["phm_objective"], dict):
                        preview = f" objective={result['phm_objective'].get('objective')}"
                    elif "readiness" in result:
                        preview = f" readiness={result['readiness']}"
                    elif "sensor_tier" in result:
                        preview = f" tier={result['sensor_tier']}"
                    elif "n_studies" in result:
                        preview = f" n_studies={result['n_studies']}"
                elif isinstance(result, list):
                    preview = f" n_items={len(result)}"
                print(f"  [OK]   {tool_name}: {brief}{preview}")
        except Exception as e:
            print(f"  [CRASH] {tool_name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append((tool_name, str(e)))

    print()
    n_tools = len(build_tool_definitions([DATASET_NAME]))
    print(f"Total tools registered: {n_tools}")
    print(f"Tools tested: {len(tests)}")
    print(f"Figures registered: {len(FIGURE_REGISTRY)}")
    print()

    if failures:
        print(f"FAIL: {len(failures)} tool(s) crashed.")
        return 1

    # Spot-check a few results
    print("Spot checks:")
    print(f"  PHM objective: {parser.classify_phm_objective()['objective']}")
    val = parser.validate_dataset()
    print(f"  Validation status: {val['status']}; errors={len(val['errors'])}, warnings={len(val['warnings'])}")
    print(f"  Gate summary: {val['gate_summary']}")
    rep = parser.get_replication_gaps(first_study_title)
    if isinstance(rep, dict):
        print(f"  Replication readiness ({first_study_title}): {rep['readiness']} ({rep['score']})")
        print(f"  Most critical missing: {rep.get('most_critical_missing')}")
    sens = parser.get_sensor_compatibility(first_study_title, first_sensor)
    if isinstance(sens, dict):
        print(f"  Sensor tier: {sens['sensor_tier']}; compatible plots: {sens['compatible_plots']}")
    print()
    print("PASS: all tools executed without crashing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
