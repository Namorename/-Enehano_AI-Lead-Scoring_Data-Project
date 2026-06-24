"""
Shared pytest fixtures for the lead-scoring test suite.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def minimal_df():
    """
    DataFrame with all raw columns that feature engineering and contracts expect.
    20 rows is enough for unit tests without slowing things down.
    """
    n = 20
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Lead_ID":                    [f"L{i:04d}" for i in range(n)],
        "IČO":                        [f"{i:08d}" for i in range(n)],
        "Company_Name":               [f"Company {i}" for i in range(n)],
        "CreatedDate":                pd.date_range("2023-01-01", periods=n, freq="7D").astype(str),
        "LastActivityDate":           pd.date_range("2023-06-01", periods=n, freq="5D").astype(str),
        "Status":                     rng.choice(["Open", "Working", "Closed"], n),
        "Converted":                  rng.integers(0, 2, n),
        "Closed_Won":                 rng.integers(0, 2, n),
        "Conversion_Probability":     rng.uniform(0, 1, n),
        "Win_Probability__c":         rng.uniform(0, 1, n),
        "Rule_Based_Score":           rng.integers(0, 101, n),
        "Rule_Based_Segment":         rng.choice(["High", "Medium", "Low"], n),
        "IČO_Duplicate_Flag":         rng.integers(0, 2, n),
        "Time_to_First_Response_h__c": rng.uniform(0, 48, n),
        "Days_Since_Last_Activity__c": rng.uniform(0, 120, n),
        "Days_in_Pipeline__c":        rng.uniform(0, 365, n),
        "Web_Interactions__c":        rng.integers(0, 50, n).astype(float),
        "Email_Opens__c":             rng.integers(0, 20, n).astype(float),
        "Email_Clicks__c":            rng.integers(0, 10, n).astype(float),
        "Email_Open_Rate__c":         rng.uniform(0, 1, n),
        "Form_Submissions__c":        rng.integers(0, 5, n).astype(float),
        "Content_Downloads__c":       rng.integers(0, 5, n).astype(float),
        "Chatbot_Interactions__c":    rng.integers(0, 5, n).astype(float),
        "Meetings_Held__c":           rng.integers(0, 5, n).astype(float),
        "Calls_Made__c":              rng.integers(0, 10, n).astype(float),
        "Demo_Requested__c":          rng.integers(0, 2, n).astype(int),
        "LinkedIn_Viewed__c":         rng.integers(0, 2, n).astype(int),
        "Proposal_Sent__c":           rng.integers(0, 2, n).astype(int),
        "Annual_Revenue_MCZK__c":     rng.uniform(0, 1000, n),
        "NumberOfEmployees":          rng.integers(1, 500, n).astype(float),
        "Company_Age_Years":          rng.uniform(0, 50, n),
        "Rating":                     rng.choice(["Hot", "Warm", "Cold"], n),
        "CZ_NACE_Section":            rng.choice(["J", "K", "M", "C"], n),
        "CZ_NACE_Section_Label":      ["Tech" for _ in range(n)],
        "Industry":                   ["Technology" for _ in range(n)],
        "Region_Name":                rng.choice(["Prague", "Brno", "Ostrava"], n),
        "Region_Code":                rng.choice(["CZ010", "CZ064", "CZ080"], n),
        "LeadSource":                 rng.choice(["Web", "Partner", "Event", "Cold Call"], n),
        "Legal_Form_Code":            rng.choice(["SRO", "AS", "OSVC"], n),
        "Legal_Form_Label":           ["s.r.o." for _ in range(n)],
        "Size_Band":                  rng.choice(["Small", "Medium", "Large"], n),
        "Owner":                      ["rep@example.com" for _ in range(n)],
        "Founding_Year":              rng.integers(1970, 2020, n),
    })