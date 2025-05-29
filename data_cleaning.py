import re
import pandas as pd
import numpy as np

def validate_dataframe(df, rename_map, clean_email, clean_phone, clean_contact_phone, clean_contact_email, clean_url, clean_state, clean_address):
    """
    Validate a DataFrame with specific columns and return a clean DataFrame
    with only valid records. Standardizes phone numbers to +1 format.
    
    Expected columns:
    - Company Name: Must not be empty
    - Address: Must contain address-like information (street, road, ave, etc.)
    - State: Must be a valid 2-letter US state abbreviation
    - Phone number: Must be a valid US phone number (standardized to +1 format)
    - Email: Must be a valid email address
    - Contact Name: Should have at least first and last name
    - Contact Email: Must be a valid email address
    - URL: Should be a valid URL if present
    """
    # Rename columns for easier access
    
    # Only include columns that are present in df
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Create a mask to track valid rows
    valid_rows = np.ones(len(df), dtype=bool)
    
    # Define validation functions for each column
    def validate_company_name(x):
        return isinstance(x, str) and len(x.strip()) > 0
    
    def validate_address(x):
        if pd.isna(x) or x == '':
            return True
        if not isinstance(x, str) or len(x.strip()) < 5:
            return False
        # Check for common address components
        address_patterns = [
            r'\d+\s+[A-Za-z0-9\s\.]+\s+(St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Way|Pl|Place|Ct|Court)',
            r'P\.?O\.?\s*Box\s+\d+',
            r'(Suite|Ste|Unit|Apt|Apartment)\s+\d+',
            r'\d+\s+[A-Za-z0-9\s\.]+',  # More lenient pattern for addresses without street type
        ]
        return any(re.search(pattern, x, re.IGNORECASE) for pattern in address_patterns)
    
    def validate_state(x):
        if pd.isna(x) or x == '':
            return True
        valid_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
        if not isinstance(x, str):
            return False
        return x.strip().upper() in valid_states
    
    def validate_and_standardize_phone(x):
        if pd.isna(x) or x == '':
            return True, None
        if not isinstance(x, str):
            return False, None
            
        # Remove any non-numeric prefix text like "Cell -"
        cleaned = re.sub(r'^[a-zA-Z\s\-:]+', '', x.strip())
        
        # Extract all digits from the string
        digits = re.sub(r'\D', '', cleaned)
        
        # Check if we have enough digits for a valid US phone number
        if len(digits) < 10:
            return False, None
            
        # Handle US numbers with country code
        if len(digits) == 11 and digits.startswith('1'):
            # Already has country code, just format it
            formatted = f"+{digits[0]}{digits[1:4]}{digits[4:7]}{digits[7:]}"
            return True, formatted
        elif len(digits) == 10:
            # Standard 10-digit US number, add +1 prefix
            formatted = f"+1{digits[0:3]}{digits[3:6]}{digits[6:]}"
            return True, formatted
        elif len(digits) > 11:
            # Might have an extension or be an invalid number
            # For simplicity, we'll just take the first 10 digits if the number is too long
            formatted = f"+1{digits[0:3]}{digits[3:6]}{digits[6:10]}"
            return True, formatted
        else:
            return False, None
    
    def validate_email(x):
        if pd.isna(x) or x == '':
            return True
        if not isinstance(x, str):
            return False
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, x.strip()))
    
    def validate_contact_name(x):
        if not isinstance(x, str) or len(x.strip()) < 3:
            return False
        # Check if there are at least 2 words (first and last name)
        parts = x.strip().split()
        return len(parts) >= 2
    
    def validate_url(x):
        if pd.isna(x) or x == '':
            return True
        if not isinstance(x, str):
            return False
        url_pattern = r'^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        return bool(re.match(url_pattern, x.strip()))
    
    if 'Address' in df.columns and clean_address:
        mask = df['Address'].apply(validate_address)
        valid_rows &= mask
        # Mark invalid addresses
        clean_df.loc[~mask, 'Address'] = np.nan
    
    if 'State' in df.columns and clean_state:
        valid_rows &= df['State'].apply(validate_state)
    
    if 'Company phone' in df.columns and clean_phone:
        # Apply the phone validation and standardization
        phone_results = df['Company phone'].apply(validate_and_standardize_phone)
        
        # Extract validation results and standardized numbers
        phone_valid = phone_results.apply(lambda x: x[0])
        standardized_phones = phone_results.apply(lambda x: x[1] if x[0] else np.nan)
        
        # Update the valid_rows mask
        valid_rows &= phone_valid
        
        # Update the phone numbers with standardized format
        clean_df.loc[phone_valid, 'Company phone'] = standardized_phones[phone_valid]
        clean_df.loc[~phone_valid, 'Company phone'] = np.nan
    
    if 'Contact Phone' in df.columns and clean_contact_phone:
        # Apply the phone validation and standardization
        phone_results = df['Contact Phone'].apply(validate_and_standardize_phone)
        
        # Extract validation results and standardized numbers
        phone_valid = phone_results.apply(lambda x: x[0])
        standardized_phones = phone_results.apply(lambda x: x[1] if x[0] else np.nan)
        
        # Update the valid_rows mask
        valid_rows &= phone_valid
        
        # Update the phone numbers with standardized format
        clean_df.loc[phone_valid, 'Contact Phone'] = standardized_phones[phone_valid]
        clean_df.loc[~phone_valid, 'Contact Phone'] = np.nan
    
    if 'Email' in df.columns and clean_email:
        valid_rows &= df['Email'].apply(validate_email)
    
    if 'Contact Email' in df.columns and clean_contact_email:
        valid_rows &= df['Contact Email'].apply(validate_email)
    
    if 'url' in df.columns and clean_url:
        valid_rows &= df['url'].apply(validate_url)
    # Rename back
    rename_map = {v: k for k, v in rename_map.items()}
    clean_df = clean_df.rename(columns=rename_map)
    # Return only valid rows
    return clean_df[valid_rows]

def main():
    # Create sample data (you'd replace this with your actual data loading)
    df = pd.read_excel('initial_data/master_list.xlsx', sheet_name='all')
    
    clean_df = validate_dataframe(df)
    # clean_df.to_excel('final_data/cleaned_master_list.xlsx', index=False)
    
    # Show validation statistics
    print(f"\nOriginal row count: {len(df)}")
    print(f"Cleaned row count: {len(clean_df)}")
    print(f"Removed {len(df) - len(clean_df)} invalid records")
