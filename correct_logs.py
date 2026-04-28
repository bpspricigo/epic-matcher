from jira import JIRA
import pandas as pd
import sys
import os

import json
import jirahelper

jirahelper.set_use_primary_jira(True)
jira = jirahelper.get_jira()
EPIC_LINK_FIELD = 'customfield_10109'  # Configure for your Jira instance

# Specific Jira keys to inspect
jira_keys = [
    'NAVBUG-910409',
    'NAVBUG-899693',
    'HWBUG-381695',
    'NAVBUG-915624',
    'NAVBUG-915174',
    'NAVBUG-921737',
    'NAVBUG-933578',
    'NAVBUG-933554',
    'NAVBUG-942448',
    'NAVBUG-942445',
    'NAVBUG-942438',
    'NAVBUG-943007',
    'NAVBUG-943004',
    'NAVBUG-944651',
    'NAVBUG-944505',
    'NAVBUG-947482',
    'HWBUG-389545',
    'NAVBUG-951611',
    'NAVBUG-951569',
    'NAVBUG-951382',
    'NAVBUG-951841',
    'NAVBUG-951815',
    'NAVBUG-951812',
    'NAVBUG-951796',
    'NAVBUG-951785',
    'NAVBUG-951238',
    'NAVBUG-939937',
    'NAVBUG-777967',
    'HWBUG-389737',
    'HWBUG-384556',
    'NAVBUG-954984',
    'NAVBUG-954426',
    'NAVBUG-953169',
    'HWBUG-388756',
    'NAVBUG-957937',
    'NAVBUG-824964',
    'NAVBUG-959965',
    'HWBUG-389898',
    'NAVBUG-960404',
    'NAVBUG-960391',
    'NAVBUG-944177',
    'NAVBUG-961537',
    'NAVBUG-961362',
    'NAVBUG-857380',
    'NAVBUG-961644',
    'HWBUG-389957',
    'NAVBUG-962885',
    'NAVBUG-966094',
    'NAVBUG-960406',
    'NAVBUG-953480',
    'NAVBUG-914700',
    'NAVBUG-968191',
    'NAVBUG-929166',
    'NAVBUG-968319',
    'HWBUG-389101',
    'NAVBUG-957502',
    'NAVBUG-935472',
    'HWBUG-390241',
]

# Create JQL query
jql = f'key in ({", ".join(jira_keys)})'

print(f"Fetching {len(jira_keys)} issues with epic history...")

# Fetch issues with changelog
issues = jira.search_issues(jql, maxResults=100, expand='changelog', fields=f'{EPIC_LINK_FIELD}')

print(f"Found {len(issues)} issues")

all_epics_per_issue = {}

for issue in issues:
    epics = set()

    # Current epic
    current_epic = getattr(issue.fields, EPIC_LINK_FIELD, None)
    if current_epic:
        epics.add(current_epic)

    # Past epics from changelog
    if hasattr(issue, 'changelog') and issue.changelog:
        for history in issue.changelog.histories:
            for item in history.items:
                if item.field == 'Epic Link':
                    if item.fromString:
                        epics.add(item.fromString)
                    if item.toString:
                        epics.add(item.toString)

    all_epics_per_issue[issue.key] = list(epics)

# Convert to DataFrame
df_all_epics = pd.DataFrame([
    {'Jira Key': key, 'All Epics': ', '.join(epics) if epics else 'No Epic'}
    for key, epics in all_epics_per_issue.items()
])

print("\nAll Epics (Current + Past):")
print(df_all_epics)

# Optional: Show count of epics per issue
df_all_epics['Epic Count'] = df_all_epics['All Epics'].apply(lambda x: len(x.split(', ')) if x != 'No Epic' else 0)
print("\nWith Epic Count:")
print(df_all_epics)
