import sqlite3
import pandas as pd
import random
from configs.root_paths import SQL_DB

# Step 1: Connect to the existing database
conn = sqlite3.connect(SQL_DB)

# Step 2: Load all base prompts
df = pd.read_sql_query("SELECT * FROM base_prompts", conn)

# Step 3: Filter prompts containing both "guide" and "third-grade"
filtered_df = df[
    df['base_prompt_string'].str.contains('guide', case=False, na=False) &
    df['base_prompt_string'].str.contains('third-grade', case=False, na=False)
]

# Step 4: Shuffle the filtered DataFrame (keeping original bp_idx)
filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

print(f"🎯 Total filtered prompts: {len(filtered_df)}")

# Step 5: Overwrite the base_prompts table with filtered, shuffled prompts
conn.execute("DROP TABLE IF EXISTS base_prompts")

# Recreate the table with the same schema
conn.execute("""
CREATE TABLE base_prompts (
    bp_idx INTEGER PRIMARY KEY,
    base_prompt_string TEXT
)
""")

# Insert filtered and shuffled prompts
filtered_df.to_sql("base_prompts", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("✅ Successfully overwrote base_prompts table with filtered and shuffled prompts.")
