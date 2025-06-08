import pandas as pd
from difflib import SequenceMatcher
import re
from itertools import product
import streamlit as st

def clean_text(text):
    """Clean text by removing special characters and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def remove_common_suffixes(name):
    """Remove common business suffixes from company names."""
    common_suffixes = [
        'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 
        'incorporated', 'company', 'co', 'properties', 'property',
        'holdings', 'group', 'international', 'enterprises', 'solutions',
        'services', 'technologies', 'tech', 'industries', 'systems',
        'association', 'associates', 'partners', 'consultants', 'consulting'
    ]
    
    name = clean_text(name)
    for suffix in common_suffixes:
        # Remove the suffix if it's a standalone word at the end or with punctuation
        name = re.sub(r'\b' + suffix + r'\b$', '', name)
        name = re.sub(r'\b' + suffix + r'\s', ' ', name)
    
    return name.strip()

def sequence_similarity(str1, str2):
    """Calculate sequence similarity using SequenceMatcher."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    
    str1 = clean_text(str1)
    str2 = clean_text(str2)
    
    if not str1 or not str2:
        return 0.0
        
    return SequenceMatcher(None, str1, str2).ratio()

def token_order_similarity(str1, str2):
    """Calculate similarity based on order of words and common tokens."""
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    
    str1 = clean_text(str1)
    str2 = clean_text(str2)
    
    if not str1 or not str2:
        return 0.0
    
    tokens1 = str1.split()
    tokens2 = str2.split()
    
    # If either is empty after tokenization
    if not tokens1 or not tokens2:
        return 0.0
    
    # Calculate Jaccard similarity (common tokens)
    common_tokens = set(tokens1).intersection(set(tokens2))
    all_tokens = set(tokens1).union(set(tokens2))
    jaccard = len(common_tokens) / len(all_tokens) if all_tokens else 0
    
    # Calculate order similarity
    # This weighs tokens that appear in the same position more heavily
    order_similarity = 0
    min_len = min(len(tokens1), len(tokens2))
    
    for i in range(min_len):
        if tokens1[i] == tokens2[i]:
            order_similarity += 1
    
    order_score = order_similarity / min_len if min_len > 0 else 0
    
    # Combine with weight (60% jaccard, 40% order)
    return 0.6 * jaccard + 0.4 * order_score

def company_name_similarity(name1, name2):
    """Calculate similarity between company names, handling common variations."""
    if not isinstance(name1, str) or not isinstance(name2, str):
        return 0.0
    
    # Clean and normalize names
    clean_name1 = remove_common_suffixes(name1)
    clean_name2 = remove_common_suffixes(name2)
    
    # If names are empty after cleaning
    if not clean_name1 or not clean_name2:
        return sequence_similarity(name1, name2)
    
    # Combine sequence and token order similarity
    seq_sim = sequence_similarity(clean_name1, clean_name2)
    token_sim = token_order_similarity(clean_name1, clean_name2)
    
    return 0.5 * seq_sim + 0.5 * token_sim

def url_similarity(url1, url2):
    """Calculate similarity between website URLs."""
    if not isinstance(url1, str) or not isinstance(url2, str):
        return 0.0
    
    # Extract domain from URLs
    url1 = url1.lower().strip()
    url2 = url2.lower().strip()
    
    # Remove http://, https://, www. prefixes
    url1 = re.sub(r'^(https?://)?(www\.)?', '', url1)
    url2 = re.sub(r'^(https?://)?(www\.)?', '', url2)
    
    # Remove trailing slashes and everything after the domain
    url1 = re.sub(r'/.*$', '', url1)
    url2 = re.sub(r'/.*$', '', url2)
    
    if not url1 or not url2:
        return 0.0
    
    # If domains are exactly the same
    if url1 == url2:
        return 1.0
    
    return sequence_similarity(url1, url2)

def address_similarity(addr1, addr2):
    """Calculate similarity between addresses."""
    if not isinstance(addr1, str) or not isinstance(addr2, str):
        return 0.0
    
    # Normalize addresses
    addr1 = clean_text(addr1)
    addr2 = clean_text(addr2)
    
    if not addr1 or not addr2:
        return 0.0
    
    # Replace common abbreviations
    abbrev_map = {
        'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
        'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
        'apt': 'apartment', 'ste': 'suite', 'n': 'north', 's': 'south',
        'e': 'east', 'w': 'west', 'fl': 'floor'
    }
    
    for abbr, full in abbrev_map.items():
        addr1 = re.sub(r'\b' + abbr + r'\b', full, addr1)
        addr2 = re.sub(r'\b' + abbr + r'\b', full, addr2)
    
    # Combine sequence and token order similarity
    seq_sim = sequence_similarity(addr1, addr2)
    token_sim = token_order_similarity(addr1, addr2)
    
    return 0.6 * seq_sim + 0.4 * token_sim

def contact_name_similarity(name1, name2):
    """Calculate similarity between contact names."""
    if not isinstance(name1, str) or not isinstance(name2, str):
        return 0.0
    
    name1 = clean_text(name1)
    name2 = clean_text(name2)
    
    if not name1 or not name2:
        return 0.0
    
    # Split into first and last name
    parts1 = name1.split()
    parts2 = name2.split()
    
    # If only one name is provided
    if len(parts1) == 1 or len(parts2) == 1:
        return sequence_similarity(name1, name2)
    
    # Compare first names
    first_sim = sequence_similarity(parts1[0], parts2[0])
    
    # Compare last names
    last_sim = sequence_similarity(parts1[-1], parts2[-1])
    
    # Weight last name similarity higher (60% last name, 40% first name)
    return 0.4 * first_sim + 0.6 * last_sim

def state_similarity(state1, state2):
    """Calculate similarity between state values."""
    if not isinstance(state1, str) or not isinstance(state2, str):
        return 0.0
    
    state1 = clean_text(state1)
    state2 = clean_text(state2)
    
    if not state1 or not state2:
        return 0.0
    
    # If states are exactly the same
    if state1 == state2:
        return 1.0
    
    # Check if one is abbreviation of the other
    state_abbrev_map = {
        'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
        'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
        'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
        'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
        'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
        'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
        'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
        'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
        'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
        'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
        'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
        'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
        'wisconsin': 'wi', 'wyoming': 'wy'
    }
    
    # Check abbreviation matches
    if state1 in state_abbrev_map and state_abbrev_map[state1] == state2:
        return 1.0
    if state2 in state_abbrev_map and state_abbrev_map[state2] == state1:
        return 1.0
    
    # Reverse map for abbreviation to full name
    abbrev_to_full = {v: k for k, v in state_abbrev_map.items()}
    
    # Check if both are abbreviations and match to same full name
    if state1 in abbrev_to_full and state2 in abbrev_to_full:
        if abbrev_to_full[state1] == abbrev_to_full[state2]:
            return 1.0
    
    return sequence_similarity(state1, state2)

def calculate_cross_similarity(row1, row2, columns, weights=None):
    """Calculate weighted similarity between rows from different dataframes."""
    if weights is None:
        # Default weights for each column type
        weights = {
            'company_name': 0.35,
            'website_url': 0.25,
            'address': 0.15,
            'state': 0.15,
            'contact_name': 0.10
        }
    
    similarity_funcs = {
        'company_name': company_name_similarity,
        'website_url': url_similarity,
        'address': address_similarity,
        'state': state_similarity,
        'contact_name': contact_name_similarity
    }
    
    scores = {}
    total_weight = 0
    
    for col, col_type in columns.items():
        if col_type in similarity_funcs and col_type in weights:
            val1 = row1[col] if col in row1 and not pd.isna(row1[col]) else ""
            val2 = row2[col] if col in row2 and not pd.isna(row2[col]) else ""
            
            if not isinstance(val1, str):
                val1 = str(val1) if val1 else ""
            if not isinstance(val2, str):
                val2 = str(val2) if val2 else ""
            
            # Skip if both values are empty
            if not val1 and not val2:
                continue
                
            sim_func = similarity_funcs.get(col_type, sequence_similarity)
            sim_score = sim_func(val1, val2)
            
            scores[col] = sim_score
            total_weight += weights[col_type]
    
    # Calculate weighted average if we have any scores
    if scores and total_weight > 0:
        weighted_sum = sum(scores[col] * weights[columns[col]] for col in scores)
        return weighted_sum / total_weight, scores
    
    return 0.0, {}

def find_cross_dataframe_duplicates(df1, df2, columns_config, threshold=0.8, weights=None):
    """
    Find potential duplicates between two dataframes.
    
    Args:
        df1: First DataFrame (will be updated)
        df2: Second DataFrame (source of truth)
        columns_config: Dict mapping column names to their types 
        threshold: Minimum similarity score to consider as duplicate
        weights: Optional dict specifying weights for different column types
        
    Returns:
        DataFrame with potential cross-dataframe duplicates and their similarity scores
    """
    results = []
    
    # Compare each row in df1 with each row in df2
    # progress_bar = st.progress(0)
    for idx1, row1 in df1.iterrows():
        for idx2, row2 in df2.iterrows():
            # Update progress bar
            # progress = ((idx1 * len(df2) + idx2 + 1) / (len(df1) * len(df2)))
            # progress_bar.progress(progress)
            overall_score, column_scores = calculate_cross_similarity(
                row1, row2, columns_config, weights
            )
            
            if overall_score >= threshold:
                result = {
                    'df1_index': idx1,
                    'df2_index': idx2,
                    'df1_company': row1.get(next((col for col, type_ in columns_config.items() 
                                          if type_ == 'company_name'), None), ""),
                    'df2_company': row2.get(next((col for col, type_ in columns_config.items() 
                                          if type_ == 'company_name'), None), ""),
                    'similarity_score': overall_score
                }
                
                # Add individual column scores
                for col in column_scores:
                    result[f'{col}_similarity'] = column_scores[col]
                    
                results.append(result)
    # progress_bar.progress(100)

    # Create and return the results dataframe
    if results:
        results_df = pd.DataFrame(results).sort_values(by='similarity_score', ascending=False)
        
        # Keep only the best match for each df1 record
        results_df = results_df.drop_duplicates(subset=['df1_index'], keep='first')
        
        return results_df
    else:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['df1_index', 'df2_index', 'df1_company', 'df2_company', 'similarity_score'])

def update_df1_with_df2_values(df1, df2, duplicates_df, column_config, update_strategy=None):
    """
    Update df1 records with values from df2 based on identified duplicate pairs.
    
    Args:
        df1: DataFrame to be updated
        df2: Source DataFrame with preferred values
        duplicates_df: DataFrame with duplicate pairs (output from find_cross_dataframe_duplicates)
        column_config: Dict mapping column names to their types
        update_strategy: Dict specifying how to update each column type
            Possible strategies: 'df2_priority', 'df1_priority', 'longest', 'most_complete'
            
    Returns:
        Updated df1 DataFrame
    """
    if duplicates_df.empty:
        print("No duplicates found to update.")
        return df1.copy()
    
    # Make a copy of df1 to avoid modifying the original
    result_df = df1.copy()
    
    # Default update strategies for different column types if not specified
    default_strategy = {
        'company_name': 'df2_priority',     # Prefer df2 company name
        'website_url': 'df2_priority',      # Prefer df2 URL
        'address': 'most_complete',         # Use the most complete address
        'state': 'df2_priority',            # Prefer df2 state
        'contact_name': 'most_complete'     # Use the most complete contact name
    }
    
    if update_strategy is None:
        update_strategy = default_strategy
    else:
        # Fill in missing strategies with defaults
        for col_type, strategy in default_strategy.items():
            if col_type not in update_strategy:
                update_strategy[col_type] = strategy
    
    # Function to choose value based on strategy
    def choose_value(val1, val2, strategy, col_type):
        # Handle NaN values
        val1_empty = pd.isna(val1) or val1 == ""
        val2_empty = pd.isna(val2) or val2 == ""
        
        if val1_empty and val2_empty:
            return ""
        elif val1_empty:
            return val2
        elif val2_empty:
            return val1
        
        if strategy == 'df2_priority':
            return val2
        elif strategy == 'df1_priority':
            return val1
        elif strategy == 'longest':
            return val2 if len(str(val2)) > len(str(val1)) else val1
        elif strategy == 'most_complete':
            if col_type == 'address':
                # For addresses, use the one with more tokens
                return val2 if len(str(val2).split()) > len(str(val1).split()) else val1
            elif col_type == 'contact_name':
                # For contact names, prefer full names over initials
                val1_has_dots = '.' in str(val1)
                val2_has_dots = '.' in str(val2)
                if val1_has_dots and not val2_has_dots:
                    return val2
                elif val2_has_dots and not val1_has_dots:
                    return val1
                else:
                    return val2 if len(str(val2)) > len(str(val1)) else val1
            else:
                # Default to longest for other types
                return val2 if len(str(val2)) > len(str(val1)) else val1
        else:
            return val2  # Default to df2 value if strategy not recognized
    
    # Process each duplicate pair
    updates_made = 0
    
    # progress_bar_2 = st.progress(0)
    for _, dup_row in duplicates_df.iterrows():

        # progress_bar_2.progress((updates_made + 1) / len(duplicates_df))

        df1_idx = dup_row['df1_index']
        df2_idx = dup_row['df2_index']
        
        # Get the corresponding rows
        df1_row = df1.loc[df1_idx]
        df2_row = df2.loc[df2_idx]
        
        # Update each column in df1 based on the strategy
        for col in result_df.columns:
            if col in df2.columns:  # Only update if column exists in df2
                col_type = column_config.get(col)
                if col_type is None:
                    # If column type not specified, use df2_priority strategy
                    strategy = 'df2_priority'
                else:
                    strategy = update_strategy.get(col_type, 'df2_priority')
                
                # Choose the appropriate value
                new_value = choose_value(df1_row[col], df2_row[col], strategy, col_type)
                
                # Update the value if it's different
                if str(result_df.at[df1_idx, col]) != str(new_value):
                    result_df.at[df1_idx, col] = new_value
                    updates_made += 1
    
    # progress_bar.progress(100)
    return result_df

def cross_dataframe_duplicate_manager(df1, df2, threshold=0.8):
    """
    Main function to find and update duplicates between two dataframes.
    
    Args:
        df1: DataFrame to be updated (target)
        df2: DataFrame with preferred values (source)
        column_config: Dict mapping column names to their types
        threshold: Minimum similarity score to consider as duplicate
        weights: Optional dict specifying weights for different column types
        update_strategy: Dict specifying how to update each column type
        
    Returns:
        Tuple: (updated_df1, duplicates_found_df, summary_stats)
    """
    column_config = {
        'Company Name': 'company_name',
        'Company Domain Name': 'website_url',
        'Contact Full name': 'contact_name'
    }
    
    # Define custom weights (optional)
    weights = {
        'company_name': 0.25,
        'website_url': 0.35,
        'contact_name': 0.1
    }
    
    # Define update strategy (optional)
    update_strategy = {
        'company_name': 'df2_priority',
        'website_url': 'df2_priority',
        'contact_name': 'most_complete'
    }
    
    # Find cross-dataframe duplicates
    duplicates = find_cross_dataframe_duplicates(df1, df2, column_config, threshold, weights)
    
    if not duplicates.empty:
        
        updated_df1 = update_df1_with_df2_values(df1, df2, duplicates, column_config, update_strategy)
        
        # Summary statistics
        summary_stats = {
            'df1_original_count': len(df1),
            'df2_count': len(df2),
            'duplicates_found': len(duplicates),
            'df1_final_count': len(updated_df1),
            'similarity_threshold': threshold
        }
        
        return updated_df1, duplicates, summary_stats
    else:
        print("No duplicates found with the given threshold.")
        summary_stats = {
            'df1_original_count': len(df1),
            'df2_count': len(df2),
            'duplicates_found': 0,
            'df1_final_count': len(df1),
            'similarity_threshold': threshold
        }
        return df1.copy(), duplicates, summary_stats