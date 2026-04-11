"""
config.py — Shared constants for the KronoDroid temporal drift study.
"""
import os

BASE_DIR       = r"C:\Users\Nate\Desktop\krono2"
CSV_FILE       = os.path.join(BASE_DIR, "kronodroid-2021-emu-v1.csv")
LIFECYCLE_FILE = os.path.join(BASE_DIR, "permission_lifecycle_v3.csv")

LABEL_COL = "Malware"
DATE_COL  = "FirstModDate"
MIN_YEAR  = 2008

DROP_COLS = ["Package", "sha256", "MalFamily", "Detection_Ratio",
             "Scanners", "LastModDate"]

TRAIN_YEARS       = list(range(2009, 2013))   # 2009-2012
TEST_YEARS        = [2013, 2014, 2015, 2016]
AUT_EXCLUDE_YEARS = [2016]

TTL_SENTINEL = 99

LIFECYCLE_YEAR_COLS = ["intro_year", "restrict_year",
                       "deprecate_year", "announced_restriction_year"]

TEMPORAL_COLS = [
    "perm_age_mean", "perm_age_max", "perm_age_min",
    "perm_restricted_ratio",  "perm_deprecated_ratio",
    "perm_near_restrict_ratio",
    "perm_risk_restrict_mean",  "perm_risk_restrict_min",
    "perm_risk_deprecate_mean", "perm_risk_deprecate_min",
    "perm_announced_restrict_ratio",
    "perm_risk_announced_restrict_mean", "perm_risk_announced_restrict_min",
    "perm_restricted_count", "perm_near_restrict_count", "perm_deprecated_count",
    "perm_has_any_restricted", "perm_has_any_near_restrict",
    "perm_worst_age", "perm_risk_worst_restrict", "perm_worst_perm_age_at_restrict",
    "perm_age_x_near_restrict", "perm_worst_age_x_risk_restrict",
    "perm_age_std", "perm_restrict_density", "perm_unrestricted_old_ratio",
    # --- new features ---
    "perm_announced_deprecate_ratio",
    "perm_risk_announced_deprecate_mean", "perm_risk_announced_deprecate_min",
    "perm_risk_deprecate_max", "perm_risk_deprecate_std",
    "perm_new_perm_ratio", "perm_age_relative_max",
    # --- data-driven features ---
    "perm_count",
    "perm_will_restrict_count", "perm_will_restrict_ratio",
    "perm_ever_lifecycle_count", "perm_ever_lifecycle_ratio",
    "perm_count_x_risk_deprecate", "perm_count_x_will_restrict_ratio",
]

METRICS = ["f1_weighted", "f1_macro", "bal_acc"]

# High-drift static flags identified by analyze_flag_drift.py
# These have lifecycle events AND above-median gap shift (train→test)
HIGH_DRIFT_FLAGS = [
    "GET_TASKS", "READ_CONTACTS", "ACCESS_COARSE_LOCATION",
    "SYSTEM_ALERT_WINDOW", "ACCESS_WIFI_STATE", "CALL_PHONE",
    "RECEIVE_SMS", "WRITE_SETTINGS", "READ_SMS", "ACCESS_FINE_LOCATION",
    "GET_ACCOUNTS", "READ_LOGS", "CHANGE_NETWORK_STATE", "READ_CALL_LOG",
    "CHANGE_WIFI_STATE", "WRITE_CONTACTS", "INTERNET", "BROADCAST_SMS",
    "CAMERA", "PROCESS_OUTGOING_CALLS", "BROADCAST_WAP_PUSH",
    "READ_PHONE_STATE", "READ_EXTERNAL_STORAGE", "RECORD_AUDIO",
    "WRITE_CALL_LOG", "CHANGE_CONFIGURATION", "BROADCAST_STICKY",
    "WRITE_EXTERNAL_STORAGE", "RESTART_PACKAGES", "SEND_SMS",
    "USE_FINGERPRINT", "RECEIVE_MMS", "RECEIVE_WAP_PUSH", "READ_CALENDAR",
]

PER_FLAG_RISK_COLS = [f"pfr_{p}" for p in HIGH_DRIFT_FLAGS]

# All 34 PFR columns are included — the RF determines which carry signal.
# No cherry-picking to avoid overfitting to a specific dataset split.
