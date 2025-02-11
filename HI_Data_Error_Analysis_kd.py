import pandas as pd
import numpy as np

# Stefan-Boltzmann constant
sigma = 5.670374419e-8  # W·m⁻²·K⁻⁴
H = 8500  # scale height in meters

# Model parameters
a1, a2, m1, m2, n1 = 0.6146529863959505, 1.4394884523236038, 0, 0, 0

# Solar constant (W/m²)
G_on = 1353  # updated to the new solar constant

longwave_urls = {
    '001': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/001HI.2024-10-07.csv',
    '002': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/002HI.2024-10-07.csv',
    '003': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/003HI.2024-10-13.csv',
    '004': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/004HI.2024-10-13.csv',
    '005': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/005HI.2024-10-13.csv',
    '006': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/006HI.2024-10-13.csv',
    '007': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/007HI.2024-10-13.csv',
    '008': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/008HI.2024-10-13.csv',
    '009': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/009HI.2024-10-13.csv',
    '010': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/010HI.2024-10-13.csv',
    '011': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/011HI.2024-10-13.csv',
    '012': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/012HI.2024-10-13.csv',
    '013': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/013HI.2024-10-13.csv',
    '014': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/014HI.2024-10-13.csv',
    '015': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/015HI.2024-10-13.csv',
    '016': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/016HI.2024-10-13.csv',
    '017': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/017HI.2024-10-07.csv',
    '018': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/018HI.2024-10-20.csv',
    '019': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/019HI.2024-10-20.csv',
    '020': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/020HI.2024-10-20.csv',
    '021': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/021HI.2024-10-20.csv',
    '022': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/022HI.2024-10-20.csv',
    '023': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/023HI.2024-05-16.csv',
    '024': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/024HI.2024-10-20.csv',
    '025': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/025HI.2024-10-20.csv',
    '026': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/026HI.2024-10-20.csv',
    '027': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/027HI.2024-10-20.csv',
    '028': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/028HI.2024-09-03.csv',
    '029': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/029HI.2024-10-20.csv',
    '030': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/030HI.2024-10-20.csv',
    '031': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/031HI.2024-10-20.csv',
    '032': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/032HI.2024-11-10.csv',
    '033': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/033HI.2024-11-10.csv',
    '034': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/034HI.2024-11-10.csv',
    '035': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/035HI.2024-11-10.csv',
    '036': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/036HI.2024-11-10.csv',
    '037': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/037HI.2024-11-10.csv',
    '038': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/024HI.2024-11-10.csv',
    '039': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/039HI.2024-11-10.csv',
    '040': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/040HI.2024-11-10.csv',
    '041': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/041HI.2024-11-10.csv',
    '042': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/042HI.2024-11-10.csv',
    '043': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/043HI.2024-06-16.csv',
    '044': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/044HI.2024-06-16.csv',
    '045': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/045HI.2024-06-16.csv',
    '046': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/046HI.2024-11-18.csv',
    '047': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/047HI.2024-11-18.csv',
    '048': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/048HI.2024-11-18.csv',
    '049': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/049HI.2024-11-18.csv',
    '050': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/050HI.2024-11-18.csv',
    '051': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/051HI.2024-11-18.csv',
    '052': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/052HI.2024-11-18.csv',
    '053': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/053HI.2024-11-18.csv',
    '054': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/054HI.2024-11-18.csv',
    '055': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/055HI.2024-11-18.csv',
    '056': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/056HI.2024-11-18.csv',
    '057': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/057HI.2024-11-18.csv',
    '058': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/058HI.2024-11-18.csv',
    '059': 'https://raw.githubusercontent.com/akarsh1207/Longwave-Radiation-Model-Hawaii/main/Raw%20Station%20Datasets/059HI.2024-11-18.csv'
}

# Load NSRDB data (calculate datetime)
def load_nsrdb_data(file_path):
    try:
        print(f"🔍 Reading NSRDB data file: {file_path}")
        df = pd.read_csv(file_path, skiprows=2)
        print(f"✅ NSRDB data loaded successfully, shape: {df.shape}")
        print("📄 First few rows of NSRDB data:")
        print(df.head())

        # Get altitude from the second row, ninth column of the shortwave data
        altitude = df.iloc[1, 8]  # Index 1, 8 for second row, ninth column
        print(f"🌍 Site altitude: {altitude} meters")

        # Check if required columns exist
        required_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'DHI', 'GHI', 'DNI', 'Clearsky DNI', 'Solar Zenith Angle']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ NSRDB data missing the following required columns: {missing_columns}")
            return None, altitude  # Return altitude

        # Calculate datetime for NSRDB
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
        print("📅 Generated datetime column:")
        print(df['datetime'].head())

        # Check if there are any NaT in datetime column
        num_nat = df['datetime'].isna().sum()
        if num_nat > 0:
            print(f"⚠️ {num_nat} datetime values failed to parse (NaT)")
            df = df.dropna(subset=['datetime']).copy()  # Create copy to avoid SettingWithCopyWarning
            print(f"📉 Shape of NSRDB data after removing NaT: {df.shape}")

        return df, altitude  # Return dataframe and altitude

    except Exception as e:
        print(f"⚠️ Failed to read NSRDB data: {e}")
        return None, None  # Return None if error occurs

# Load longwave radiation data
def load_longwave_data(station):
    try:
        file_path = longwave_urls.get(station)
        if not file_path:
            print(f"⚠️ Longwave radiation data URL not found for station {station}")
            return None

        print(f"🔍 Reading longwave radiation data file: {file_path}")
        df = pd.read_csv(file_path, skiprows=10, low_memory=False)
        print(f"✅ Longwave radiation data loaded successfully, shape: {df.shape}")
        print("📄 First few rows of longwave radiation data:")
        print(df.head())

        # Check for header row (assuming header row has NaN in Station_ID)
        if df['Station_ID'].isna().iloc[0]:
            print("⚠️ Detected header row, deleting the first row")
            df = df.drop(index=0).reset_index(drop=True).copy()  # Create copy to avoid SettingWithCopyWarning
            print(f"📉 Shape of longwave radiation data after removing header row: {df.shape}")
            print("📄 First few rows of longwave radiation data (after removing header):")
            print(df.head())

        # Check if required columns exist
        required_columns = ['Date_Time', 'incoming_radiation_lw_set_1', 'air_temp_set_1', 'relative_humidity_set_1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ Longwave radiation data missing the following required columns: {missing_columns}")
            return None

        # Try to parse datetime column and ignore rows that cannot be parsed
        df['datetime'] = pd.to_datetime(df['Date_Time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
        print("📅 Generated datetime column:")
        print(df['datetime'].head())

        # Remove NaT from datetime column
        num_nat = df['datetime'].isna().sum()
        if num_nat > 0:
            print(f"⚠️ {num_nat} datetime values failed to parse (NaT), they will be removed")
            df = df.dropna(subset=['datetime']).copy()  # Create copy to avoid SettingWithCopyWarning
            print(f"📉 Shape of longwave radiation data after removing NaT: {df.shape}")

        # Force conversion of key columns to numeric and handle non-numeric entries
        for col in ['incoming_radiation_lw_set_1', 'air_temp_set_1', 'relative_humidity_set_1']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows containing NaN in key columns
        initial_shape = df.shape
        df = df.dropna(subset=['incoming_radiation_lw_set_1', 'air_temp_set_1', 'relative_humidity_set_1']).copy()  # Create copy to avoid SettingWithCopyWarning
        print(f"📉 Dropped rows with NaN in key columns: from {initial_shape} to {df.shape}")

        return df
    except Exception as e:
        print(f"⚠️ Failed to read longwave radiation data: {e}")
        return None

# Data alignment
def align_data(nsrdb_data, longwave_data, tolerance_minutes=1):
    try:
        print(f"🔗 Aligning data with a tolerance of {tolerance_minutes} minutes")
        merged_df = pd.merge_asof(
            nsrdb_data.sort_values('datetime'),
            longwave_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(minutes=tolerance_minutes)
        ).copy()  # Create copy to avoid SettingWithCopyWarning
        print(f"✅ Data alignment completed, merged shape: {merged_df.shape}")
        print("📄 First few rows of merged data:")
        print(merged_df.head())

        # Check how many NaNs are in the merged data
        num_nan = merged_df[['incoming_radiation_lw_set_1', 'air_temp_set_1', 'relative_humidity_set_1']].isna().sum()
        print(f"⚠️ NaN values in merged data:\n{num_nan}")

        return merged_df
    except Exception as e:
        print(f"⚠️ Data alignment failed: {e}")
        return None

# Error analysis
def analyze_error(merged_data, station, altitude):
    results = []
    if merged_data is not None and not merged_data.empty:
        print("📊 Starting error analysis")

        # Apply additional filter conditions
        condition_1 = (merged_data['GHI'] > 0) & (merged_data['DHI'] > 0)
        condition_2 = (merged_data['DNI'] / merged_data['Clearsky DNI'] > 0) & (
                    merged_data['DNI'] / merged_data['Clearsky DNI'] <= 1.5)
        condition_3 = (merged_data['air_temp_set_1'] <= 90) & (merged_data['air_temp_set_1'] >= -80)
        condition_4 = (merged_data['incoming_radiation_lw_set_1'] > 0)
        condition_5 = (merged_data['Solar Zenith Angle'] < 72.5)
        condition_6 = (merged_data['GHI'] < (
                    1.2 * G_on * np.cos(np.radians(merged_data['Solar Zenith Angle'])) ** 1.2 + 50))
        condition_7 = (merged_data['DNI'] < (
                    0.95 * G_on * np.cos(np.radians(merged_data['Solar Zenith Angle'])) ** 0.2 + 10))

        # Apply all filter conditions and create a copy to avoid SettingWithCopyWarning
        merged_data = merged_data[
            condition_1 & condition_2 & condition_3 & condition_4 & condition_5 & condition_6 & condition_7].copy()
        print(f"📉 Shape of data after filtering: {merged_data.shape}")

        # Handle NaN or invalid values
        merged_data = merged_data.dropna(subset=['air_temp_set_1', 'relative_humidity_set_1']).copy()
        print(f"📉 Shape after removing NaN: {merged_data.shape}")

        if merged_data.empty:
            print("⚠️ All rows were removed after deleting NaN")
            return results

        # Check for invalid values and convert them to NaN
        key_columns = ['DHI', 'GHI', 'DNI', 'Clearsky DNI', 'incoming_radiation_lw_set_1']
        for col in key_columns:
            if not np.issubdtype(merged_data[col].dtype, np.number):
                print(f"⚠️ Column {col} is not numeric, attempting to convert")
                merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

            if not np.isfinite(merged_data[col]).all():
                print(f"⚠️ Column {col} contains infinite or invalid values, setting these rows to NaN")
                merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

        # Remove rows containing NaN in key columns
        merged_data = merged_data.dropna(subset=key_columns).copy()
        print(f"📉 Shape after removing NaN in key columns: {merged_data.shape}")

        if merged_data.empty:
            print("⚠️ All rows were removed after deleting NaN in key columns")
            return results

        # Calculate k_d, k_t, cf
        merged_data['k_d'] = merged_data['DHI'] / merged_data['GHI']
        merged_data['k_t'] = merged_data['DNI'] / merged_data['Clearsky DNI']
        merged_data['cf'] = a1 * (merged_data['k_d'] ** a2) + m1 * (merged_data['k_t'] ** m2) + n1

        # Calculate clear-sky emissivity
        temp = merged_data['air_temp_set_1']
        rh = merged_data['relative_humidity_set_1']

        try:
            e_clear_sky = (
                    0.6 +
                    1.652 * np.sqrt(
                (6.112 * np.exp(17.625 * temp / (temp - 30.11 + 273.15)) * rh / 100) / 1013.25
            ) +
                    0.15 * (np.exp(-altitude / H) - 1)
            )
            merged_data['e_clear_sky'] = e_clear_sky
            print("✅ e_clear_sky calculation successful")
        except Exception as e:
            print(f"⚠️ Failed to calculate e_clear_sky: {e}")
            merged_data['e_clear_sky'] = np.nan

        # Handle NaN or invalid values
        merged_data['e_clear_sky'] = merged_data['e_clear_sky'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(
            lower=0)

        # Calculate predicted emissivity (e_sky_pred)
        merged_data['e_sky_pred'] = (1 - merged_data['cf']) * merged_data['e_clear_sky'] + merged_data['cf']

        # Calculate actual emissivity
        merged_data['epsilon_actual'] = merged_data['incoming_radiation_lw_set_1'] / (
                    sigma * ((merged_data['air_temp_set_1'] + 273.15) ** 4))
        merged_data['epsilon_actual'] = merged_data['epsilon_actual'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate error
        merged_data['error'] = merged_data['e_sky_pred'] - merged_data['epsilon_actual']

        # Calculate MBE, RMSE, rMBE, rRMSE
        def relative_mbe(x, mean_actual):
            return x.mean() / mean_actual if mean_actual != 0 else np.nan

        def relative_rmse(x, mean_actual):
            return np.sqrt(np.mean(x ** 2)) / mean_actual if mean_actual != 0 else np.nan

        yearly_errors = merged_data.groupby(merged_data['datetime'].dt.year).agg(
            MBE=('error', 'mean'),
            RMSE=('error', lambda x: np.sqrt(np.mean(x ** 2))),
            rMBE=('error', lambda x: relative_mbe(x, merged_data.loc[x.index, 'epsilon_actual'].mean())),
            rRMSE=('error', lambda x: relative_rmse(x, merged_data.loc[x.index, 'epsilon_actual'].mean()))
        ).reset_index().rename(columns={'datetime': 'year'})

        print("📈 Yearly error statistics:")
        print(yearly_errors)

        for _, row in yearly_errors.iterrows():
            results.append({
                'year': int(row['year']),
                'MBE': row['MBE'],
                'RMSE': row['RMSE'],
                'rMBE': row['rMBE'],
                'rRMSE': row['rRMSE'],
                'station': station  # Ensure station information is included
            })

    else:
        print("❌ Merged data is empty, cannot perform error analysis")

    return results


# **Main program**
def main():
    stations = [f'{i:03d}' for i in range(1, 60)]  # Generate station list 001HI to 059HI

    all_results = []
    missing_data = []

    for station in stations:
        print(f"\n🔍 Processing data for station {station}...")

        # Update NSRDB data URL for the station
        nsrdb_file_path = f'https://raw.githubusercontent.com/Joe-pooh/NSRDB_HI_DATA/refs/heads/main/{station}HI_2023.csv'

        # Read data
        nsrdb_data, altitude = load_nsrdb_data(nsrdb_file_path)
        longwave_data = load_longwave_data(station)

        if nsrdb_data is None or longwave_data is None:
            missing_data.append(station)
            print(f"❌ Data for station {station} is missing or failed to load, skipping station")
            continue

        # Data alignment
        merged_data = align_data(nsrdb_data, longwave_data, tolerance_minutes=5)  # Can try increasing tolerance

        if merged_data is None or merged_data.empty:
            print(f"❌ No valid data after aligning for station {station}!")
            missing_data.append(station)  # Record missing data stations
            continue

        # Error analysis
        error_results = analyze_error(merged_data, station, altitude)

        # If error results are empty, print warning
        if error_results:
            all_results.extend(error_results)

    # Process results
    results_df = pd.DataFrame(all_results)

    # Add missing data flag
    results_df['missing_sw_data'] = results_df['station'].apply(lambda x: 'Yes' if x in missing_data else 'No')

    # Ensure all stations are included in results, fill missing stations
    missing_stations_df = pd.DataFrame({
        'station': missing_data,
        'missing_sw_data': ['Yes'] * len(missing_data),
    })

    # Combine both parts of the data
    final_results_df = pd.concat([results_df, missing_stations_df], ignore_index=True)

    # Save Excel file
    output_excel_path = r'C:\Users\jiw181\PycharmProjects\pythonProject1\2025\error_results_all_stations_only_k_d.xlsx'
    try:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # Write all data to one sheet
            final_results_df.to_excel(writer, sheet_name='All_Stations', index=False)
        print(f"\n✅ Error analysis results for all stations saved to '{output_excel_path}'")
    except Exception as e:
        print(f"\n❌ Error saving Excel file: {e}")


if __name__ == "__main__":
    main()
