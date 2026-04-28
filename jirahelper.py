import pytz
from jira import JIRA, JIRAError
from termcolor import colored
from datetime import datetime, timezone
import authhelper
import os
import re

from config import jira as _jira_cfg

SECONDARY_JIRA_URL = _jira_cfg['secondary_url']
PRIMARY_JIRA_URL  = _jira_cfg['primary_url']

jira_instance = None
JIRA_MAX_RESULTS = 1000
USE_PRIMARY_JIRA = False
JIRAHELPER_RESULT_OK = True


def set_resultOK(value):
    global JIRAHELPER_RESULT_OK
    JIRAHELPER_RESULT_OK = value


def get_resultOK():
    return JIRAHELPER_RESULT_OK


def set_use_primary_jira(use_primary_jira: bool):
    global USE_PRIMARY_JIRA
    USE_PRIMARY_JIRA = use_primary_jira


def use_primary_jira():
    return USE_PRIMARY_JIRA


def get_jira():
    global jira_instance
    global JIRAHELPER_RESULT_OK

    if not jira_instance:
        try:
            if USE_PRIMARY_JIRA:
                print(f"Using jira url: {PRIMARY_JIRA_URL}")
                jira_instance = JIRA(token_auth=authhelper.get_primary_token(), server=PRIMARY_JIRA_URL, max_retries=1)
            else:
                print(f"Using jira url: {SECONDARY_JIRA_URL}")
                jira_instance = JIRA(server=SECONDARY_JIRA_URL, basic_auth=authhelper.get_secondary_auth())

        except Exception as ex:
            JIRAHELPER_RESULT_OK = False
            print(f"Error during connecting to JIRA URL\nDetails: {str(ex)}")
            raise Exception(f"Unable to connect to JIRA URL")

    return jira_instance


def get_tickets_for_analysis_based_on_environment_filter(data):
    full_filter = '(' + data.filter + ') and (labels is EMPTY or labels != ' + data.analyzed_label + ")"
    tickets = get_jira().search_issues(full_filter, maxResults=JIRA_MAX_RESULTS)

    if len(tickets) >= JIRA_MAX_RESULTS:
        raise RuntimeError(f"WARNING! Too many tickets found ({len(tickets)}) for pre analysis, cannot process them")

    ticket_list = []
    for ticket in tickets:
        ticket_list.append(str(ticket.key))
    print(f"Found {str(len(tickets))} tickets for query: '{full_filter}'")
    return ticket_list


def upload_comment_data(issue_string, pluginName, comment=None):
    jira = get_jira()
    jira_issue = jira.issue(issue_string)

    # submit comment
    if comment:
        comment_prefix = _jira_cfg['comment_prefix'] + " " + pluginName
        for jira_comment in jira_issue.fields.comment.comments:
            if jira_comment.body.startswith(comment_prefix):
                delete_jira_comment_safe(jira_comment)

        create_jira_comment_safe(jira, issue_string, comment_prefix, comment)


def create_jira_comment_safe(jira: JIRA, issue_string: str, comment_prefix: str, comment: str):
    global JIRAHELPER_RESULT_OK

    try:
        jira_issue = jira.issue(issue_string)
        if USE_PRIMARY_JIRA:
            jira.add_comment(jira_issue, comment_prefix + '\n\n' + comment)
 # primary Jira uses team member role instead of developers role
 # visibility={'type': 'role', 'value': 'Team Member'})
        else:
            jira.add_comment(jira_issue, comment_prefix + '\n\n' + comment,
                             visibility={'type': 'role', 'value': 'Developers'})
    except Exception as e:
        JIRAHELPER_RESULT_OK = False
        print(colored(f"Could not add Comment to Jira: {e}", "yellow"))


def delete_jira_comment_safe(jira_comment: JIRA.comment):
    global JIRAHELPER_RESULT_OK

    try:
        jira_comment.delete()
    except JIRAError:
        JIRAHELPER_RESULT_OK = False
        print(colored("Could not delete old tracker comment!", "yellow"))


def create_jira_attachement_safe(jira: JIRA, jira_issue: JIRA.issue, filepath):
    global JIRAHELPER_RESULT_OK

    file = os.path.basename(filepath)
    size = os.path.getsize(filepath) / 1024 / 1024
    try:
        for attachment in jira_issue.fields.attachment:
            if str(attachment) == file:
                print(f"Upload file {file}({size:.2f} MB) aborted. File already in attachments.")
                JIRAHELPER_RESULT_OK = False
                return
        jira.add_attachment(issue=jira_issue, attachment=filepath)

    except Exception as e:
        JIRAHELPER_RESULT_OK = False
        print(colored(f"Could not add new attachement {file}( {size:.2f} MB) to Jira: {e}", "yellow"))


def delete_jira_attachement_safe(jira: JIRA, attachement_id: str):
    global JIRAHELPER_RESULT_OK

    try:
        jira.delete_attachment(attachement_id)
    except JIRAError:
        JIRAHELPER_RESULT_OK = False
        print(colored("Could not delete old Attachement!", "yellow"))


def update_jira_comment_safe(jira_comment: JIRA.comment, comment_prefix: str, comment: str):
    global JIRAHELPER_RESULT_OK

    try:
        jira_comment.update(body=comment_prefix + '\n\n' + comment)
    except Exception as e:
        JIRAHELPER_RESULT_OK = False
        print(colored(f"Could not update Jira Comment: {e}", "yellow"))


def update_jira_label_safe(jira_issue: JIRA.issue):
    global JIRAHELPER_RESULT_OK

    try:
        jira_issue.update(fields={"labels": jira_issue.fields.labels})
    except Exception as e:
        JIRAHELPER_RESULT_OK = False
        print(colored(f"Error trying to update jira label: {e}"))


def upload_add_attachment(issue_string, file):
    jira = get_jira()
    jira_issue = jira.issue(issue_string)
    create_jira_attachement_safe(jira, jira_issue, file)


def upload_label_data(issue_string, data, log_path, pluginName, comment=None):
    jira = get_jira()
    jira_issue = jira.issue(issue_string)

    # append analyzed label
    # This must be the last step, since otherwise if script gets interrupted after setting the label but before e.g.
    # uploading files, we will miss the files. If we miss the label, then in worst case we will reupload the files and
    # post the same comment again, which is safer.
    labels = jira_issue.fields.labels
    if data and not data.analyzed_label in labels:
        labels.append(data.analyzed_label)
        update_jira_label_safe(jira_issue)


def upload_domain(issue_string, newDomain, newSubDomain=None):
    global JIRAHELPER_RESULT_OK

    jira = get_jira()
    jira_issue = jira.issue(issue_string)

    # https://confluence.atlassian.com/jirakb/how-to-find-any-custom-field-s-ids-744522503.html
    # These are the custom fields for Domain and Subdomain. The update call only worked with the custom field name.
    try:
        set_domain_field(jira_issue, newDomain)
        set_sub_domain_field(jira_issue, newSubDomain)
    except Exception as e:
        JIRAHELPER_RESULT_OK = False
        print(f"failed to set domain: {e}")


def get_domain_history(issue_string):
    domains = []
    jira = get_jira()
    jira_issue = jira.issue(issue_string, expand='changelog')

    for history_item in jira_issue.changelog.histories:
        for item in history_item.items:
            if item.field == "Domain":
                domains.append(item.fromString)
                domains.append(item.toString)
    return domains


def upload_analysis_data(issue_string, data, issue_directory, comment=None):
    jira = get_jira()
    jira_issue = jira.issue(issue_string)

    should_retry = comment == 'FAIL -> missing logs'
    add_first_tracker_comment_flag = True
    ignore_time_frame_s = 60 * 60 * 4
    server_timezone = _jira_cfg['server_timezone']

    if len(comment) > 60000:
        print(colored("comment too long to upload ", "red"))
    # submit comment
    elif comment and data.post_comment:
        comment_prefix = _jira_cfg['comment_prefix']

        if data.comment_prefix:
            comment_prefix = data.comment_prefix

        if should_retry and data.comment_prefix:
            comment_prefix += "_retry"

        for jira_comment in jira_issue.fields.comment.comments:
            if jira_comment.body.startswith(comment_prefix):
                add_first_tracker_comment_flag = False

        if add_first_tracker_comment_flag:
            create_jira_comment_safe(jira, issue_string, comment_prefix, comment)

    # upload files
    if data.file_upload_filter:
        filter_patterns = []
        for filter_pattern in data.file_upload_filter:
            filter_patterns.append(re.compile(filter_pattern))
        for root, dir_names, file_names in os.walk(issue_directory):
            for file_name in file_names:
                for filter_pattern in filter_patterns:
                    if filter_pattern.match(file_name):
                        if os.stat(os.path.join(root, file_name)).st_size > 0:
                            print("Uploading: " + root + "/" + file_name)
                            create_jira_attachement_safe(jira, jira_issue, os.path.join(root, file_name))
                        else:
                            print("Ignoring empty file: " + root + "/" + file_name)

    # append analyzed label
    # This must be the last step, since otherwise if script gets interrupted after setting the label but before e.g.
    # uploading files, we will miss the files. If we miss the label, then in worst case we will reupload the files and
    # post the same comment again, which is safer.

    if data.comment_prefix:
        comment_prefix = data.comment_prefix + "_retry"

    tracker_comment = [
        jira_comment
        for jira_comment in jira_issue.fields.comment.comments
        if jira_comment.body.startswith(comment_prefix)
    ]

    success_comment = [
        jira_comment
        for jira_comment in jira_issue.fields.comment.comments
        if jira_comment.body.startswith(data.comment_prefix)
    ]

    success_comment_exists = len(success_comment) > 0

    labels = jira_issue.fields.labels
    successfully_processed_once = data.analyzed_label in labels
    already_retried = len(tracker_comment) > 0

    processing_message = "label_processed_successfully"

    if not successfully_processed_once:
        if not should_retry and already_retried:  ### All files present, but required a retry attempt
            labels.append(data.analyzed_label)
            update_jira_label_safe(jira_issue)
            update_jira_comment_safe(tracker_comment[0], data.comment_prefix, comment)

        elif not should_retry and success_comment_exists:  ### Update existing success comment
            labels.append(data.analyzed_label)
            update_jira_label_safe(jira_issue)
            delete_jira_comment_safe(success_comment[0])
            create_jira_comment_safe(jira, issue_string, comment_prefix, comment)

        elif not should_retry:  ### All files present on first processing, label as success
            labels.append(data.analyzed_label)
            update_jira_label_safe(jira_issue)

        elif already_retried:  ### Already retried and @ignore_time_frame_s has passed, label as processed but could not analyse files
            tracker_comment_time = datetime.strptime(tracker_comment[0].created.split('.')[0], '%Y-%m-%dT%H:%M:%S')
            server_tz = pytz.timezone(server_timezone)
            current_date_server_time = datetime.now(server_tz).strftime('%Y-%m-%dT%H:%M:%S')
            current_date_server_time = datetime.strptime(current_date_server_time, '%Y-%m-%dT%H:%M:%S')

            if time_delta_over_threshold(tracker_comment_time, current_date_server_time, ignore_time_frame_s):
                labels.append(data.analyzed_label)
                update_jira_label_safe(jira_issue)
                update_jira_comment_safe(tracker_comment[0], data.comment_prefix, " Tracker analysed multiple times but no logs found")

            processing_message = "label_processing_retry_failed"

        else:
            processing_message = "label_processing_do_retry"

    print(f"PROCESSING MESSAGE {processing_message}")
    return processing_message


def time_delta_over_threshold(timestamp_one, timestamp_two, threshold_s):
    return (timestamp_two - timestamp_one).total_seconds() >= threshold_s


def get_integrated_field(ticket):
    if use_primary_jira():
        return str(ticket.fields.customfield_10812)
    else:
        return str(ticket.fields.customfield_10521)


def get_pre_integrated_field(ticket):
    if use_primary_jira():
        return str(ticket.fields.customfield_11202)
    else:
        return str(ticket.fields.customfield_15700)


def set_domain_field(ticket, newDomain):
    if use_primary_jira():
        ticket.update(fields={'customfield_10300': {'value': newDomain}})
    else:
        ticket.update(fields={'customfield_10105': {'value': newDomain}})


def set_sub_domain_field(ticket, newSubDomain=None):
    if newSubDomain is not None:
        if use_primary_jira():
            ticket.update(fields={'customfield_10801': {'value': newSubDomain}})
        else:
            ticket.update(fields={'customfield_13137': {'value': newSubDomain}})
