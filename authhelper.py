import os
import netrc
import re
from getpass import getpass
from sys import platform

from config import jira as _jira_cfg

primary_host   = _jira_cfg['primary_host']
secondary_host = _jira_cfg['secondary_host']

_primary_token    = ""
_secondary_user   = ""
_secondary_pass   = ""


def _netrc_path() -> str:
    prefix = "_" if platform.startswith("win") else "."
    return os.path.join(os.path.expanduser("~"), f"{prefix}netrc")


def get_primary_token() -> str:
    """Return the API token for the primary Jira instance (read from .netrc)."""
    global _primary_token
    if not _primary_token:
        try:
            _primary_token = netrc.netrc(_netrc_path()).hosts[primary_host][2]
        except Exception:
            _primary_token = os.environ.get('JIRA_PRIMARY_TOKEN', '')
    return _primary_token


def get_secondary_auth() -> tuple:
    """Return (username, password) for the secondary Jira instance (read from .netrc)."""
    global _secondary_user, _secondary_pass
    if not _secondary_user or not _secondary_pass:
        try:
            entry = netrc.netrc(_netrc_path()).hosts[secondary_host]
            _secondary_user = entry[0]
            _secondary_pass = entry[2]
        except Exception:
            _secondary_user = os.environ.get('JIRA_SECONDARY_USER', '')
            _secondary_pass = os.environ.get('JIRA_SECONDARY_PASS', '')
    return (_secondary_user, _secondary_pass)
