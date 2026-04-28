#!/usr/bin/env python3
"""
Step 1: Fetch all issues from the navigation epics.
Saves the result as JSON for model training.

Usage:
    python3 scripts/fetch_epic_issues.py

Output:
    scripts/epic_training_data.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import jirahelper
from config import epics as EPICS, jira as _jira_cfg

jirahelper.set_use_primary_jira(True)
jira = jirahelper.get_jira()

EPIC_LINK_FIELD = 'customfield_10109'  # Configure for your Jira instance

all_data = []
total = 0

for epic_key, epic_name in EPICS.items():
    _types = ', '.join(f'"{t}"' for t in _jira_cfg['issue_types'])
    jql = f'"Epic Link" = {epic_key} AND issuetype in ({_types}) ORDER BY updated DESC'
    print(f"\n{'='*60}")
    print(f"Epic: {epic_key} ({epic_name})")

    issues = []
    start_at = 0
    while True:
        batch = jira.search_issues(jql, startAt=start_at, maxResults=100,
                                   fields=f'summary,description,{EPIC_LINK_FIELD},status,project')
        if not batch:
            break
        issues.extend(batch)
        start_at += len(batch)
        if len(batch) < 100:
            break

    print(f"  Found: {len(issues)} issues")

    for issue in issues:
        all_data.append({
            "key": issue.key,
            "summary": issue.fields.summary or "",
            "description": issue.fields.description or "",
            "epic_key": epic_key,
            "epic_name": epic_name,
            "status": str(issue.fields.status),
            "project": str(issue.fields.project),
        })

    total += len(issues)

print(f"\n{'='*60}")
print(f"TOTAL issues fetched: {total}")

OUTPUT = os.path.join(os.path.dirname(__file__), 'epic_training_data.json')
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"Saved to: {OUTPUT}")

print(f"\n{'Epic Key':<16} {'Epic Name':<35} {'Count'}")
print("-" * 60)
for epic_key, epic_name in sorted(EPICS.items(), key=lambda x: x[1]):
    count = sum(1 for d in all_data if d['epic_key'] == epic_key)
    print(f"{epic_key:<16} {epic_name:<35} {count}")
