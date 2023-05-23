import pandas as pd
import random as rd

rows = 1_000_000
# Create a DataFrame with one column containing random numbers from -1000 to 1000 and an empty 'yies' column
df = pd.DataFrame(
    {'A': [rd.randint(-1000, 1000) for _ in range(rows)],
     'B': [rd.randint(-1000, 1000) for _ in range(rows)],
     'C': [rd.randint(-1000, 1000) for _ in range(rows)],
     'D': [rd.randint(-1000, 1000) for _ in range(rows)],  
     'E': [''] * rows}
    )

# Save the DataFrame as a CSV file
df.to_csv('x.csv', index=False)