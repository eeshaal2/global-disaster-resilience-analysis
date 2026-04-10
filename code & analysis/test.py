import pandas as pd

def describe_dataset(file_path="final_resilience_data.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please run the processing script first.")
        return

    print("="*60)
    print(f"DATASET REPORT: {file_path}")
    print("="*60)

    # 1. Basic Structure
    print(f"\n[1] SHAPE: {df.shape[0]} Rows, {df.shape[1]} Columns")
    
    print("\n[2] COLUMN DATA TYPES:")
    print(df.dtypes)

    # 2. Preview
    print("\n[3] DATA PREVIEW (First 5 Rows):")
    print(df.head().to_string())

    # 3. Missing Values
    print("\n[4] MISSING VALUE ANALYSIS:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_report = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    # Only show columns that actually have missing data
    print(missing_report[missing_report['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

    # 4. Statistical Summary (Numerical)
    print("\n[5] STATISTICAL SUMMARY (Numerical Columns):")
    # T stands for Transpose, making it easier to read wide tables
    print(df.describe().T.to_string())

    # 5. Categorical Summary
    print("\n[6] CATEGORICAL SUMMARY:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        print(f" - {col}: {unique_count} unique values")
        # If few unique values, list them
        if unique_count < 10:
            print(f"   Values: {df[col].unique()}")

    # 6. Project Specific Quality Check
    print("\n[7] PROJECT METRICS CHECK:")
    # Check if we have valid calculated indices
    valid_dii = df['DII'].notna().sum()
    valid_cri = df['CRI'].notna().sum()
    print(f" - Rows with valid Disaster Impact Index (DII): {valid_dii} ({valid_dii/len(df)*100:.1f}%)")
    print(f" - Rows with valid Composite Resilience Index (CRI): {valid_cri} ({valid_cri/len(df)*100:.1f}%)")

if __name__ == "__main__":
    describe_dataset()