import pandas as pd
from googlesearch import search
import time
import csv
from typing import List, Dict
import logging

def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('company_search.log'),
            logging.StreamHandler()
        ]
    )

def read_companies(file_path: str, sheet_name, column_name = 'Company') -> List[str]:
    """
    Read company names from a CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        List[str]: List of company names
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_csv(file_path)
        return df[column_name].unique().tolist()
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return []

def search_company(company_name: str, num_results: int = 3) -> List[str]:
    """
    Search for a company name on Google and return top URLs
    
    Args:
        company_name (str): Name of the company to search
        num_results (int): Number of URLs to return
        
    Returns:
        List[str]: List of URLs found
    """
    urls = []
    try:
        # Add 'company' to the search query to get more relevant results
        search_query = f"{company_name} company"
        for url in search(search_query, num_results=num_results):
            if 'linkedin.com' in url or 'facebook.com' in url or 'wikipedia' in url:
                continue
            elif '/search?num=' in url:
                continue
            urls.append(url)
            
        # Add a delay to avoid hitting rate limits
        time.sleep(2)
    except Exception as e:
        logging.error(f"Error searching for {company_name}: {e}")
    
    return urls

def save_results(results: Dict[str, List[str]], output_file: str):
    """
    Save search results to a CSV file
    
    Args:
        results (Dict[str, List[str]]): Dictionary of company names and their URLs
        output_file (str): Path to the output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Company', 'URL 1', 'URL 2', 'URL 3'])
            
            for company, urls in results.items():
                # Pad with empty strings if less than 3 URLs found
                urls_padded = urls + [''] * (3 - len(urls))
                writer.writerow([company] + urls_padded)
                
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def main():
    """Main function to orchestrate the search process"""
    setup_logging()
    
    # Configure these paths as needed
    input_file = 'initial_data/2025 CARVC Conference Final Registration List.xlsx'
    output_file = 'final_data/2025_CARV_sponsors_urls.csv'
    column_name = 'Company'
    sheet_name = '2025 Sponsors'

    
    logging.info("Starting company URL search process")
    
    # Read companies from CSV
    companies = read_companies(input_file, sheet_name, column_name)
    if not companies:
        logging.error("No companies found in input file")
        return
    
    # Store results
    results = {}
    total_companies = len(companies)
    
    for i, company in enumerate(companies, 1):
        logging.info(f"Processing company {i}/{total_companies}: {company}")
        urls = search_company(company)
        results[company] = urls
        
        # Save results periodically in case of interruption
        if i % 10 == 0:
            save_results(results, output_file)
            
    # Save final results
    save_results(results, output_file)
    logging.info("Search process completed")

if __name__ == "__main__":
    main()