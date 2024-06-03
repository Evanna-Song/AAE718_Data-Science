### Problem1
import os
def csv_files(directory):
    csv_file_list = []

    # os.walk generates the file names in a directory tree by walking either top-down or bottom-up.
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                # os.path.join constructs a full file path.
                csv_file_list.append(os.path.join(root, file))

    return csv_file_list

directory_path = r'C:\Users\Lenovo\Desktop\emissions'
print(csv_files(directory_path))


### Problem2
import pandas as pd
def load_emission_csv(file_path, year):
    df = pd.read_csv(file_path)
    df['year'] = year
    return df

def load_emissions(directory):
    files = csv_files(directory)
    df_list = []
    for file in files:
        year = os.path.basename(file).split('.')[0]  # Assuming the filename is like '1970.csv'
        df = load_emission_csv(file, year)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

directory_path = r'C:\Users\Lenovo\Desktop\emissions'
emissions_df = load_emissions(directory_path)
print(emissions_df.head())

# Extra work I think is useful
def export_to_excel(dataframe, output_path):
    """
    This function exports a DataFrame to an Excel file.
    """
    dataframe.to_excel(output_path, index=False)

# Export the DataFrame to an Excel file
output_path = r'C:\Users\Lenovo\Desktop\emissions_combined.xlsx'
export_to_excel(emissions_df, output_path)
print(f"Data has been exported to {output_path}")



### Problem3
import pandas as pd
import os
def csv_files(directory):
    csv_file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_file_list.append(os.path.join(root, file))

    return csv_file_list

def load_emission_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def load_emissions(directory):
    files = csv_files(directory)
    df_list = []
    for file in files:
        df = load_emission_csv(file)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


directory_path = r'C:\Users\Lenovo\Desktop\emissions'
emissions_df = load_emissions(directory_path)
print(emissions_df.head())


### Problem4
def merge_emissions_with_country_codes(emissions_directory, country_code_path):
    # Load the emissions data
    emissions_df = load_emissions(emissions_directory)

    # Load the country codes data
    country_codes_df = pd.read_csv(country_code_path)

    # Merge the data frames on the appropriate columns
    merged_df = pd.merge(emissions_df, country_codes_df, left_on='Country', right_on='name', how='left')

    # Select relevant columns
    columns_to_keep = list(emissions_df.columns) + ['alpha-2', 'region', 'sub-region']
    merged_df = merged_df[columns_to_keep]

    return merged_df


# Example usage
emissions_directory_path = r'C:\Users\Lenovo\Desktop\emissions'
country_code_file_path = r'C:\Users\Lenovo\Desktop\country_codes.csv'

merged_df = merge_emissions_with_country_codes(emissions_directory_path, country_code_file_path)
print(merged_df.head())


### Problem5
def country_code_to_excel(dataframe, output_path):
    """
    This function exports a DataFrame to an Excel file.
    """
    dataframe.to_excel(output_path, index=False)

# Export the DataFrame to an Excel file
output_path = r'C:\Users\Lenovo\Desktop\country_code_combined.xlsx'
country_code_to_excel(merged_df, output_path)
print(f"Data has been exported to {output_path}")

import matplotlib.pyplot as plt

# Load the combined data
file_path = r'C:\Users\Lenovo\Desktop\country_code_combined.xlsx'
data = pd.read_excel(file_path)

# Histogram of total global CO2 emissions by region
plt.figure(figsize=(10, 6))
region_co2_sum = data.groupby('region')['Emissions.Type.CO2'].sum().sort_values()
region_co2_sum.plot(kind='barh')
plt.xlabel('Total CO2 Emissions')
plt.ylabel('Region')
plt.title('Total CO2 Emissions by Region')
plt.tight_layout()
plt.savefig('total_co2_emissions_by_region.png')
plt.show()

# Scatterplot of the ratio of CO2 emissions per capita to GDP
plt.figure(figsize=(10, 6))
plt.scatter(data['Ratio.Per Capita'], data['Ratio.Per GDP'], alpha=0.5)
plt.xlabel('CO2 Emissions per Capita')
plt.ylabel('CO2 Emissions per GDP')
plt.title('CO2 Emissions per Capita vs. CO2 Emissions per GDP')
plt.grid(True)
plt.tight_layout()
plt.savefig('co2_emissions_per_capita_vs_gdp.png')
plt.show()

# Pie charts by emission type for the Asian region
asia_data = data[data['region'] == 'Asia']
asia_emissions = asia_data[['Emissions.Type.CO2', 'Emissions.Type.N2O', 'Emissions.Type.CH4']].sum()
plt.figure(figsize=(8, 8))
asia_emissions.plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Emissions Types Distribution in Asia')
plt.tight_layout()
plt.savefig('emissions_types_distribution_asia.png')
plt.show()



### Problem6
import pandas as pd
def dirty_data(file_path, output_file_path):
    # Load the data from the CSV file
    df = pd.read_csv(file_path, header=None)

    # Extract the required columns
    order_id = df.iloc[3:, 0].dropna().values  # Skip first 3 rows and drop NaN values
    segment = df.iloc[3:, 1].dropna().values
    ship_mode = df.iloc[3:, 2].dropna().values
    sales = df.iloc[3:, 3:].stack().values  # Flatten the 2D array to 1D array

    # Ensure all arrays are of the same length
    min_length = min(len(order_id), len(segment), len(ship_mode), len(sales))
    order_id = order_id[:min_length]
    segment = segment[:min_length]
    ship_mode = ship_mode[:min_length]
    sales = sales[:min_length]

    # Create a new DataFrame with the cleaned data
    cleaned_df = pd.DataFrame({
        'order_id': order_id,
        'segment': segment,
        'ship_mode': ship_mode,
        'sales': sales
    })

    # Drop any rows with aggregate data
    cleaned_df = cleaned_df[~cleaned_df['order_id'].str.contains('Total|Grand', na=False)]

    # Write the cleaned data to a new CSV file
    cleaned_df.to_csv(output_file_path, index=False)

    return cleaned_df

# Test
file_path = r'C:\Users\Lenovo\Desktop\dirty_data_01.csv'
output_file_path = r'C:\Users\Lenovo\Desktop\cleaned_data.csv'
cleaned_df = dirty_data(file_path, output_file_path)
print(cleaned_df.head())



### Problem7
import pandas as pd
def school_data(ussd_file_path: str, layout_file_path: str) -> pd.DataFrame:
    # Define column names and locations based on layout files
    columns = [
        ('FIPS State Codes', 1, 2),
        ('District ID', 4, 8),
        ('District name', 10, 81),
        ('Total Population', 83, 90),
        ('Population aged 5-17', 92, 99),
        ('Number of children in poverty', 101, 108)
    ]

    col_names = [col[0] for col in columns]
    col_positions = [(col[1] - 1, col[2]) for col in columns]

    # Read the ussd20.txt file with the specified encoding
    data = pd.read_fwf(
        ussd_file_path,
        colspecs=col_positions,
        names=col_names,
        encoding='latin1'  # Adjusting encoding to handle unicode errors
    )

    # Converting FIPS State Codes to State Names
    fips_to_state = {
        '01': 'Alabama',
        '02': 'Alaska',
        '04': 'Arizona',
        '05': 'Arkansas',
        '06': 'California',
    }
    data['state'] = data['FIPS State Codes'].apply(lambda x: fips_to_state.get(x, 'NA'))

    return data


#  Test
ussd_file_path = r'C:\Users\Lenovo\Desktop\school_data\school_data\ussd20.txt'
layout_file_path = r'C:\Users\Lenovo\Desktop\school_data\school_data\2020-district-layout.txt'
df = school_data(ussd_file_path, layout_file_path)
print(df.head())


