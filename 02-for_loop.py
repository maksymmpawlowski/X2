import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('x.csv')

# Iterate through each row
for index, row in df.iterrows():
    # Extract values from columns A, B, C, and D
    A = row['A']
    B = row['B']
    C = row['C']
    D = row['D']

    # Calculate the value for column E based on the formula
    E = (A * B) - (C * D)

    # Update the value in column E for the current row
    df.at[index, 'E'] = E

    # Save the updated DataFrame as a CSV file
    df.to_csv('x.csv', index=False)
