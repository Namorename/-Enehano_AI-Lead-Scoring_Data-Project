"""
data_generator.py
=================
Final version optimized for specific Czech RES export headers.
Generates 'lead_train.csv' for machine learning training.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

# --- Configuration ---
INPUT_FILE = "res_open_data_sample.csv"
OUTPUT_FILE = "lead_train.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_leads():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    print(f"Reading {INPUT_FILE}...")
    # Czech exports often use comma or semicolon
    try:
        df_res = pd.read_csv(INPUT_FILE, encoding='utf-8')
        if len(df_res.columns) < 5: raise Exception("Delimiter error")
    except:
        df_res = pd.read_csv(INPUT_FILE, encoding='utf-8', sep=';')

    # Mapping based on the headers you provided
    # Standardizing names to match train.py expectations
    leads = []
    current_year = datetime.now().year

    print("Synthesizing CRM behavior and aligning features...")

    # We use a sample of 30k rows to ensure a robust model
    sample_size = min(30000, len(df_res))
    df_sample = df_res.sample(sample_size, random_state=RANDOM_SEED)

    for idx, row in df_sample.iterrows():
        # --- 1. Real Data Extraction (Firmographics) ---
        # Using exact headers from your export
        raw_ico = str(row.get('IČO', '00000000'))
        ico = raw_ico.split('.')[0].zfill(8) # Clean decimals and pad to 8 digits
        
        company_name = str(row.get('Obchodní jméno/název', 'Unknown Entity'))
        legal_form = str(row.get('Statistická právní forma (název)', 's.r.o.'))
        
        # Extract NACE section (first character of the code)
        nace_code = str(row.get('Hlavní ekonomická činnost (CZ NACE2025) (kód)', 'J'))
        nace_section = nace_code[0].upper() if nace_code and nace_code != 'nan' else "J"

        # Calculate Company Age
        origin_date = str(row.get('Datum vzniku', '2015-01-01'))
        try:
            # Assumes YYYY-MM-DD or similar where first 4 are year
            founding_year = int(origin_date[:4])
            age = max(0, current_year - founding_year)
        except:
            age = random.randint(1, 25)

        # --- 2. Synthetic CRM Telemetry (Predictive Features) ---
        ttfr = round(random.uniform(0.1, 72.0), 1) # Time to first response
        web_visits = random.randint(0, 70)
        emails_sent = random.randint(1, 15)
        opens = random.randint(0, emails_sent)
        clicks = random.randint(0, opens)
        
        # Binary behavioral signals
        demo = 1 if (web_visits > 35 and ttfr < 12.0 and random.random() > 0.4) else 0
        linkedin = 1 if random.random() > 0.7 else 0

        # --- 3. Ground Truth Conversion Logic ---
        # Probability model: Speed + Engagement = Success
        prob = 0.12
        if ttfr < 3.0: prob += 0.40      # Critical factor for CRM success
        if web_visits > 25: prob += 0.25  # Intent signal
        if demo == 1: prob += 0.20        # Strong intent
        
        # Final outcome with some statistical noise
        converted = 1 if (prob + random.uniform(-0.1, 0.1)) > 0.55 else 0

        leads.append({
            "Lead_ID": f"00Q{idx}",
            "IČO": ico,
            "Company_Name": company_name,
            "Legal_Form_Label": legal_form,
            "CZ_NACE_Section": nace_section,
            "Industry": "Technology", # Base for OHE, NACE section is the real driver
            "NumberOfEmployees": random.randint(10, 500), # Synthetic scale
            "Annual_Revenue_MCZK__c": round(random.uniform(1.0, 800.0), 1),
            "Time_to_First_Response_h__c": ttfr,
            "Web_Interactions__c": web_visits,
            "Email_Opens__c": opens,
            "Email_Clicks__c": clicks,
            "Emails_Sent__c": emails_sent,
            "Email_Open_Rate__c": round(opens/emails_sent, 2) if emails_sent > 0 else 0,
            "Form_Submissions__c": random.randint(0, 6),
            "Calls_Made__c": random.randint(0, 12),
            "Meetings_Held__c": random.randint(0, 5),
            "Content_Downloads__c": random.randint(0, 4),
            "Chatbot_Interactions__c": random.randint(0, 10),
            "Days_Since_Last_Activity__c": random.randint(0, 45),
            "Days_in_Pipeline__c": random.randint(1, 120),
            "Company_Age_Years": age,
            "Demo_Requested__c": demo,
            "Proposal_Sent__c": 1 if (demo and random.random() > 0.3) else 0,
            "LinkedIn_Viewed__c": linkedin,
            "LeadSource": random.choice(["Web", "LinkedIn", "Referral", "Cold Call"]),
            "Size_Band": random.choice(["Small", "Medium", "Large", "Enterprise"]),
            "Rating": random.choice(["Hot", "Warm", "Cold"]),
            "Converted": converted
        })

    # Save to standardized filename
    final_df = pd.DataFrame(leads)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"DONE! File saved as: {OUTPUT_FILE}")
    print(f"Rows processed: {len(final_df)}")
    print(f"Sample IČO from output: {final_df['IČO'].iloc[0]}")
    print("="*50)

if __name__ == "__main__":
    generate_leads()