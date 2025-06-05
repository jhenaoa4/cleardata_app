import pandas as pd
from difflib import SequenceMatcher
import re
from itertools import combinations

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

def calculate_overall_similarity(row1, row2, columns, weights=None):
    """Calculate weighted similarity across multiple columns."""
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

def find_duplicates(df, columns_config, threshold=0.8, weights=None):
    """
    Find potential duplicates in a dataframe based on similarity metrics.
    
    Args:
        df: Input pandas DataFrame
        columns_config: Dict mapping column names to their types 
            (e.g., {'name': 'company_name', 'url': 'website_url'})
        threshold: Minimum similarity score to consider as duplicate
        weights: Optional dict specifying weights for different column types
        
    Returns:
        DataFrame with potential duplicates and their similarity scores
    """
    results = []
    
    # Generate all unique pairs of indices
    for (idx1, row1), (idx2, row2) in combinations(df.iterrows(), 2):
        overall_score, column_scores = calculate_overall_similarity(
            row1, row2, columns_config, weights
        )
        
        if overall_score >= threshold:
            result = {
                'index1': idx1,
                'index2': idx2,
                'company1': row1.get(next((col for col, type_ in columns_config.items() 
                                      if type_ == 'company_name'), None), ""),
                'company2': row2.get(next((col for col, type_ in columns_config.items() 
                                      if type_ == 'company_name'), None), ""),
                'similarity_score': overall_score
            }
            
            # Add individual column scores
            for col in column_scores:
                result[f'{col}_similarity'] = column_scores[col]
                
            results.append(result)
    
    # Create and return the results dataframe
    if results:
        results_df = pd.DataFrame(results).sort_values(by='similarity_score', ascending=False)
        return results_df
    else:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['index1', 'index2', 'company1', 'company2', 'similarity_score'])

# Example usage:
def merge_duplicates(df, duplicates_df, column_config, merge_strategy=None):
    """
    Merge duplicate records in the original dataframe based on identified duplicate pairs.
    
    Args:
        df: Original pandas DataFrame
        duplicates_df: DataFrame with duplicate pairs (output from find_duplicates)
        column_config: Dict mapping column names to their types
        merge_strategy: Dict specifying how to merge each column type (default uses sensible strategies)
            Possible strategies: 'longest', 'first', 'last', 'most_complete', 'concatenate'
            
    Returns:
        DataFrame with duplicates merged
    """
    if duplicates_df.empty:
        print("No duplicates to merge.")
        return df.copy()
    
    # Make a copy of the original dataframe to avoid modifying it
    result_df = df.copy()
    
    # Default merge strategies for different column types if not specified
    default_strategy = {
        'company_name': 'longest',      # Use the longest company name
        'website_url': 'first',         # Use the first non-empty URL
        'address': 'most_complete',     # Use the most complete address
        'state': 'first',               # Use the first non-empty state
        'contact_name': 'most_complete' # Use the most complete contact name
    }
    
    if merge_strategy is None:
        merge_strategy = default_strategy
    else:
        # Fill in missing strategies with defaults
        for col_type, strategy in default_strategy.items():
            if col_type not in merge_strategy:
                merge_strategy[col_type] = strategy
    
    # Build a graph of connected components (groups of duplicates)
    import networkx as nx
    G = nx.Graph()
    
    # Add all indices as nodes
    for idx in result_df.index:
        G.add_node(idx)
    
    # Add edges between duplicate pairs
    for _, row in duplicates_df.iterrows():
        G.add_edge(row['index1'], row['index2'])
    
    # Find connected components (groups of duplicates)
    duplicate_groups = list(nx.connected_components(G))
    
    # Function to merge values based on strategy
    def merge_values(values, strategy, col_type):
        values = [v for v in values if not pd.isna(v) and v != ""]
        if not values:
            return ""
        
        if strategy == 'first':
            return values[0]
        elif strategy == 'last':
            return values[-1]
        elif strategy == 'longest':
            return max(values, key=len) if all(isinstance(v, str) for v in values) else values[0]
        elif strategy == 'concatenate':
            # For non-string values, convert to string first
            str_values = [str(v) if not isinstance(v, str) else v for v in values]
            # Remove duplicates while preserving order
            unique_values = []
            for v in str_values:
                if v not in unique_values:
                    unique_values.append(v)
            return " | ".join(unique_values)
        elif strategy == 'most_complete':
            if col_type == 'address':
                # For addresses, use the one with most tokens
                return max(values, key=lambda x: len(str(x).split())) if all(isinstance(v, str) for v in values) else values[0]
            elif col_type == 'contact_name':
                # For contact names, prefer full names over initials
                full_names = [v for v in values if '.' not in v]
                return full_names[0] if full_names else values[0]
            else:
                # Default to longest for other types
                return max(values, key=len) if all(isinstance(v, str) for v in values) else values[0]
        else:
            return values[0]  # Default to first value if strategy not recognized
    
    # Process each group of duplicates
    processed_indices = set()
    
    for group in duplicate_groups:
        if len(group) <= 1:
            continue  # Skip groups with only one record
        
        group = list(group)
        
        # Choose the primary record (we'll keep this one and merge others into it)
        primary_idx = group[0]
        
        # Get all records in this group
        group_records = result_df.loc[group]
        
        # For each column in the dataframe
        for col in result_df.columns:
            col_type = column_config.get(col)
            if col_type is None:
                # If column type not specified, use 'first' strategy
                strategy = 'first'
            else:
                strategy = merge_strategy.get(col_type, 'first')
            
            # Get values for this column from all records in the group
            values = group_records[col].tolist()
            
            # Merge values based on strategy
            merged_value = merge_values(values, strategy, col_type)
            
            # Update the primary record with merged value
            result_df.at[primary_idx, col] = merged_value
        
        # Mark all indices in this group as processed
        processed_indices.update(group)
    
    # Keep only one record from each duplicate group
    indices_to_keep = []
    for group in duplicate_groups:
        # Keep only the first record from each group
        indices_to_keep.append(next(iter(group)))
    
    # Add all indices that weren't part of any duplicate group
    all_indices = set(result_df.index)
    non_duplicate_indices = all_indices - processed_indices
    indices_to_keep.extend(non_duplicate_indices)
    
    # Return the deduplicated dataframe10
    return result_df.loc[indices_to_keep].reset_index(drop=True)

def dups_manage(df, id_column, company_name_col, contact_name_col, website_col):
    # df = pd.read_excel('initial_data/8. Master List SepOct.xlsx')#.head(1000)

    df.set_index(id_column, inplace=True)
    
    # Define column types for similarity calculation
    column_config = {
        'Company Name': company_name_col,
        'Website': website_col,
        'Contact name': contact_name_col
    }

    column_config = {k: v for k, v in column_config.items() if v}
    
    # Custom weights (optional)
    weights = {
        'company_name': 0.25,
        'website_url': 0.35,
        'contact_name': 0.10
    }
    
    # Find duplicates with 70% similarity threshold
    duplicates = find_duplicates(df, column_config, threshold=0.8, weights=weights)
    duplicates[['company1', 'company2', 'similarity_score']]#.to_excel('final_data/8_Master_List_duplicates.xlsx', index=False)
    
    # Print results
    if not duplicates.empty:
        print(duplicates[['company1', 'company2', 'similarity_score']].to_string(index=False))
        
        # Print detailed column similarities for first match
        first_match = duplicates.iloc[0]
        print("\nDetailed similarity for top match:")
        for col in column_config:
            if f'{col}_similarity' in duplicates.columns:
                print(f"{col}: {first_match[f'{col}_similarity']:.4f}")
        
        # Merge duplicates
        print("\nMerging duplicates...")
        
        # Define custom merge strategies (optional)
        custom_strategy = {
            'company_name': 'longest',
            'website_url': 'first',
            'address': 'most_complete',
            'state': 'first',
            'contact_name': 'longest',
            'Phone': 'first'
        }
        
        merged_df = merge_duplicates(df, duplicates, column_config, custom_strategy).drop_duplicates(subset=[company_name_col, contact_name_col, website_col], keep='first')

        # Save the merged dataframe to a CSV file
        # merged_df.to_excel('final_data/8_Master_List_deduplicated.xlsx', index=False)
        return merged_df, len(df), len(merged_df)
    else:
        return df, len(df), len(df)
        print("No duplicates found with the given threshold.")
    