# This file is dummy code to see if i get how the hpc works

# test_script.py
import time
from datetime import datetime

# Get the start time
start_time = datetime.now()
print(f"Job started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Simulate a task that runs for 10 seconds
time.sleep(10)

# Get the end time
end_time = datetime.now()
print(f"Job finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Print the duration
duration = end_time - start_time
print(f"Total duration: {duration}")
