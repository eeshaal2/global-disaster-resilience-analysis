import pandas as pd
import numpy as np

def load_emdat(path: str) -> pd.DataFrame:
    """
    Load and aggregate EM-DAT disaster data.
    """
    try:
        # Load data - 'Start Year' is the critical time column
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return pd.DataFrame()

    # Map the exact column names you provided to standard internal names
    col_map = {
        "ISO": "iso3",
        "Country": "country_disaster",
        "Region": "region",
        "Disaster Group": "disaster_group",
        "Disaster Subgroup": "disaster_subgroup",
        "Disaster Type": "disaster_type",
        "Start Year": "year",
        "Total Deaths": "total_deaths",
        "No. Injured": "injured",
        "No. Affected": "no_affected",
        "No. Homeless": "no_homeless",
        "Total Affected": "total_affected",
    }

    # Rename columns that exist in the file
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Handle Damage columns - Prioritize Adjusted, fallback to nominal
    if "Total Damage, Adjusted ('000 US$)" in df.columns:
        df["damage_raw"] = pd.to_numeric(df["Total Damage, Adjusted ('000 US$)"], errors="coerce")
    elif "Total Damage ('000 US$)" in df.columns:
        df["damage_raw"] = pd.to_numeric(df["Total Damage ('000 US$)"], errors="coerce")
    else:
        df["damage_raw"] = 0

    # Convert '000 USD to full USD
    df["total_damage_usd"] = df["damage_raw"].fillna(0) * 1000.0

    # Filter for valid rows (Must have ISO and Year)
    df = df.dropna(subset=["iso3", "year"])
    df["year"] = df["year"].astype(int)

    # Aggregate to Country-Year level
    # We sum impacts, and count the number of events
    agg_df = (
        df.groupby(["iso3", "year", "region", "country_disaster"])
        .agg(
            events_count=("iso3", "count"),
            total_deaths=("total_deaths", "sum"),
            total_affected=("total_affected", "sum"),
            total_damage_usd=("total_damage_usd", "sum")
        )
        .reset_index()
    )
    
    return agg_df

def load_worldbank_indicators(path: str) -> pd.DataFrame:
    """
    Load World Bank data, melt to long format, and pivot indicators.
    """
    try:
        wb = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return pd.DataFrame()

    # Identify year columns (e.g., "2000 [YR2000]")
    year_cols = [c for c in wb.columns if "[YR" in c]
    
    # Melt to long format
    wb_long = wb.melt(
        id_vars=["Country Name", "Country Code", "Series Name", "Series Code"],
        value_vars=year_cols,
        var_name="year_raw",
        value_name="value"
    )

    # Extract numeric year
    wb_long["year"] = wb_long["year_raw"].str.extract(r"(\d{4})").astype(float)
    wb_long = wb_long.dropna(subset=['year'])
    wb_long["year"] = wb_long["year"].astype(int)

    # Convert value to numeric (Coerce '..' to NaN)
    wb_long["value"] = pd.to_numeric(wb_long["value"], errors="coerce")

    # Filter for key indicators (using standard Series Codes)
    indicator_map = {
        "SP.POP.TOTL": "population",
        "NY.GDP.MKTP.CD": "gdp_current_usd",
        "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct" 
    }

    wb_long = wb_long[wb_long["Series Code"].isin(indicator_map.keys())]
    wb_long["indicator"] = wb_long["Series Code"].map(indicator_map)

    # Pivot: Index = Country + Year, Columns = Indicators
    wb_pivot = wb_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="indicator",
        values="value"
    ).reset_index()

    # If gdp_growth_pct is missing but GDP exists, calculate it manually
    if "gdp_growth_pct" not in wb_pivot.columns and "gdp_current_usd" in wb_pivot.columns:
        wb_pivot = wb_pivot.sort_values(["Country Code", "year"])
        # Fix FutureWarning by specifying fill_method=None
        wb_pivot["gdp_growth_pct"] = wb_pivot.groupby("Country Code")["gdp_current_usd"].pct_change(fill_method=None) * 100

    return wb_pivot

def load_hdi(path: str) -> pd.DataFrame:
    """
    Load HDI data handling the specific structure:
    Row 0: Headers (Unnamed: 1 is Country)
    Row 1: Sub-headers/Metadata
    Row 2: Start of Data
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return pd.DataFrame()

    # Identify columns based on your snippet
    # Unnamed: 1 -> Country
    # Human Development Index (HDI) -> Value
    
    rename_map = {
        "Unnamed: 1": "country_hdi",
        "Human Development Index (HDI) ": "hdi", # Note: Check for trailing space
        "Life expectancy at birth": "life_expectancy"
    }
    
    # Clean column names (strip whitespace) before renaming
    df.columns = df.columns.str.strip()
    
    # Rename known columns
    df = df.rename(columns={k.strip(): v for k, v in rename_map.items() if k.strip() in df.columns})
    
    # Fallback if specific rename didn't work (using position 1 for Country if needed)
    if "country_hdi" not in df.columns and len(df.columns) > 1:
         df.rename(columns={df.columns[1]: "country_hdi"}, inplace=True)

    # Filter only necessary columns
    cols_to_keep = ["country_hdi", "hdi", "life_expectancy"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep].copy()
    
    # --- CLEANING ROWS ---
    # 1. Drop rows where Country is NaN
    df = df.dropna(subset=["country_hdi"])
    
    # 2. Drop rows where Country column contains "Country" (the header row in body)
    df = df[df["country_hdi"].astype(str).str.strip() != "Country"]
    
    # 3. Clean Country names
    df["country_hdi"] = df["country_hdi"].astype(str).str.strip()
    
    # 4. Force Numeric HDI
    # This turns strings like "Value" or "2023" into NaN
    if "hdi" in df.columns:
        df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")
    
    # 5. Drop rows where HDI became NaN (Removes the "Value" and Year rows)
    df = df.dropna(subset=["hdi"])

    return df

def build_country_year_panel():
    print("Loading Datasets...")
    emdat = load_emdat("emat_disasters.csv")
    wb = load_worldbank_indicators("worldbank_data.csv")
    hdi = load_hdi("hdi_data.csv")

    if wb.empty:
        print("CRITICAL: World Bank data failed to load or is empty. Check filename.")
        return

    print("Merging Datasets...")
    # Merge WB (Base) with EM-DAT
    # Left join ensures we keep years with no disasters
    merged = pd.merge(
        wb, 
        emdat, 
        how="left", 
        left_on=["Country Code", "year"], 
        right_on=["iso3", "year"]
    )

    # Merge HDI
    # Note: Matches on Country Name. 
    if not hdi.empty and "country_hdi" in hdi.columns:
        merged = pd.merge(
            merged,
            hdi,
            how="left",
            left_on="Country Name",
            right_on="country_hdi"
        )

    # --- FEATURE ENGINEERING ---
    print("Calculating Indices...")
    
    # Fill NaN for disaster stats with 0 (No record = no disaster)
    fill_zeros = ["events_count", "total_deaths", "total_affected", "total_damage_usd"]
    for col in fill_zeros:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Avoid ZeroDivisionError
    merged["population_safe"] = merged["population"].replace(0, np.nan)
    merged["gdp_pc_safe"] = merged["gdp_per_capita_usd"].replace(0, np.nan)

    # 1. Normalized Impact Metrics
    merged["fatalities_per_1m"] = (merged["total_deaths"] / merged["population_safe"]) * 1_000_000
    merged["affected_pct"] = (merged["total_affected"] / merged["population_safe"]) * 100
    merged["damage_pct_gdp"] = (merged["total_damage_usd"] / merged["gdp_current_usd"]) * 100

    # 2. Disaster Impact Index (DII)
    # DII = ((Fatalities + Affected_Pop) / GDP_pc) * Severity (1)
    merged["DII"] = (merged["fatalities_per_1m"] + merged["affected_pct"]) / merged["gdp_pc_safe"]

    # 3. Composite Resilience Index (CRI)
    # CRI = Adaptive_Capacity / (Exposure * Vulnerability)
    # A = HDI, E = events_count, V = damage_pct_gdp
    
    # Ensure inputs are numeric
    if "hdi" in merged.columns:
        merged["hdi"] = pd.to_numeric(merged["hdi"], errors="coerce")
    
    # Smooth Exposure and Vulnerability
    E = merged["events_count"].replace(0, 0.1) # Low exposure
    V = merged["damage_pct_gdp"].replace(0, 0.001) # Low vulnerability
    
    merged["CRI"] = merged["hdi"] / (E * V)

    # 4. Resilience Recovery Score Proxy (RRS)
    # Proxy = (GDP Growth + HDI) / 2
    merged["gdp_growth_pct"] = pd.to_numeric(merged["gdp_growth_pct"], errors="coerce")
    merged["RRS_Proxy"] = (merged["gdp_growth_pct"] + (merged["hdi"] * 10)) / 2 

    # Cleanup
    final_df = merged.drop(columns=["population_safe", "gdp_pc_safe", "damage_raw"], errors="ignore")

    # Save
    output_file = "final_resilience_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Success! Saved {len(final_df)} rows to {output_file}")
    print("Columns included:", list(final_df.columns))

if __name__ == "__main__":
    build_country_year_panel()