import sqlite3
import pandas as pd
import os
from configs.root_paths import SQL_DB

# Path for new filtered DB
NEW_SQL_DB = SQL_DB.replace("base_prompts.db", "filtered_base_prompts.db")

# Load data from original DB
conn = sqlite3.connect(SQL_DB)
df = pd.read_sql_query("SELECT * FROM base_prompts", conn)
conn.close()

print(f"📋 Loaded {len(df)} total prompts from original DB.")

# Filter prompts containing both 'guide' and 'third-grade'
filtered_df = df[
    df['base_prompt_string'].str.contains('guide', case=False) &
    df['base_prompt_string'].str.contains('third-grade', case=False)
].copy()

print(f"📊 Found {len(filtered_df)} prompts containing both 'guide' and 'third-grade'.")

# Shuffle rows but keep bp_idx unchanged
filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to NEW DB
if os.path.exists(NEW_SQL_DB):
    os.remove(NEW_SQL_DB)

conn_new = sqlite3.connect(NEW_SQL_DB)
filtered_df.to_sql('base_prompts', conn_new, index=False, if_exists='replace')
conn_new.close()

print(f"✅ Filtered and shuffled prompts saved to {NEW_SQL_DB}. Total prompts: {len(filtered_df)}")
