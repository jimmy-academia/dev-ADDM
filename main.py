from addm.tasks.constants import ALL_TOPICS, ALL_POLICIES, GROUP_TOPICS

usage = """
ADDM - Adaptive Decision-Making with LLMs
==========================================

Run experiment:
  .venv/bin/python -m addm.tasks.cli.run_experiment --policy G1_allergy_V2 --dev --sample

Options:
  --policy POLICY   Policy ID (e.g., G1_allergy_V2) or comma-separated list
  --topic TOPIC     Run all V0-V3 for a topic (e.g., G1_allergy)
  --group GROUP     Run all policies in a group (e.g., G1)
  --dev             Dev mode (results/dev/)
  --sample          Use 3 verdict samples per policy
  -n N              Number of restaurants (default: 100)
  --k K             Reviews per restaurant (25/50/100/200)
"""

print(usage)

print("TOPICS (18):")
print("-" * 40)
for group, topics in GROUP_TOPICS.items():
    print(f"  {group}: {', '.join(t.split('_', 1)[1] for t in topics)}")

print(f"\nPOLICIES: {len(ALL_POLICIES)} total (18 topics x 4 variants V0-V3)")
