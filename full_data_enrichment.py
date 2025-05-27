import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from company_search import search_company

class DataEnrichmentSystem:
    def __init__(self, pms_list_file: str = 'Full enrichment/PMS_names.csv', gateway_list_file: str = 'Full enrichment/gateway_names.csv'):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Comprehensive phone patterns
        self.phone_patterns = [
            # US/Canada formats
            r'\+?1?\s*\(?\d{3}\)?\s*[-.]?\d{3}[-.]?\d{4}',  # (123) 456-7890 or 123-456-7890
            r'\+?1?\s*\d{3}[-.]?\d{3}[-.]?\d{4}',           # 1234567890 or 123 456 7890
            r'(?:Phone|Tel|Call|contact|office|mobile|fax|support)(?:\s*(?:us|:|\())\s*(?:\+?1?\s*[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # Fixed the closing parenthesis
            r'(?:Toll Free|toll free|Toll-Free|toll-free)[\s:]+(?:\+?1?\s*[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            # International formats
            r'\+\d{1,4}\s*\(?\d{1,4}\)?\s*\d{6,12}',        # +XX (XXX) XXXXXXXX
            r'00\d{1,4}\s*\d{6,12}'                          # 00XX XXXXXXXXXX
        ]
        
        # Comprehensive email patterns
        self.email_patterns = [
            # Basic email
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            # Common prefixes
            r'(?:mailto:|Email:|e-mail:|email us:|contact:|support:|info:|)[^\S\n]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            # HTML attributes
            r'contact@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'info@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'support@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'sales@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'admin@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        ]
        
        # Common contact page patterns
        self.contact_page_patterns = [
            '/contact', '/contact-us', '/contact_us', '/contactus',
            '/about/contact', '/get-in-touch', '/reach-us', '/support',
            '/help', '/about', '/connect'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Read PMS names from the CSV file
        with open(pms_list_file, 'r') as f:
            self.pms_systems = [line.strip() for line in f.readlines() if line.strip()]
        
        # Compilation of search patterns
        self.pms_patterns = [
            # Regex patterns to find PMS names in text
            re.compile(rf'\b{re.escape(pms)}\b', re.IGNORECASE) 
            for pms in self.pms_systems
        ]

        """
        Initialize the Payment Gateway Detector with known payment platforms
        """
        # Common payment gateways and their identifiers
        self.payment_gateways = {
            'Stripe': [
                'stripe.com',
                'stripe.js',
                'checkout.stripe.com',
                'stripe-js',
                'data-stripe',
                'stripePaymentHandler'
            ],
            'PayPal': [
                'paypal.com',
                'paypalobjects.com',
                'paypal-button',
                'paypal.Buttons',
                'data-paypal',
                'paypal-sdk'
            ],
            'Square': [
                'squareup.com',
                'square.js',
                'squarespace',
                'data-square',
                'squarePaymentForm'
            ],
            'Braintree': [
                'braintree-api',
                'braintreegateway',
                'braintree.js',
                'braintree-web'
            ],
            'Authorize.net': [
                'authorize.net',
                'authorizenet',
                'AcceptUI.js'
            ],
            'WePay': [
                'wepay.com',
                'wepay-widget'
            ],
            'Adyen': [
                'adyen.com',
                'adyenConfiguration',
                'checkout.adyen'
            ],
            'WorldPay': [
                'worldpay.com',
                'worldpayLibrary'
            ],
            'CyberSource': [
                'cybersource.com',
                'flex.cybersource'
            ]

        }

        # Read PMS names from the CSV file
        with open(gateway_list_file, 'r') as f:
            gt_systems = [line.strip() for line in f.readlines() if line.strip()]
        
        for gt in gt_systems:
            self.payment_gateways[gt] = [gt]

    def fetch_and_parse_url(self, url):
        """
        Fetches the content of the URL and parses it using BeautifulSoup.
        
        Args:
            url (str): The URL to fetch and parse.
            
        Returns:
            BeautifulSoup: Parsed HTML content.
        """
        try:
            # Add scheme if not present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Make request
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except Exception as e:
            self.logger.error(f"Error fetching and parsing {url}: {str(e)}")
            return None

    def clean_phone(self, phone):
        """Clean and standardize phone number format"""
        if not phone:
            return None
            
        # Remove all non-numeric characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Handle different formats
        if cleaned.startswith('+'):
            # International format
            if len(cleaned) >= 8:  # Minimum length for international numbers
                return cleaned
        else:
            # US/Canada format
            if len(cleaned) == 10:
                return f"+1{cleaned}"
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                return f"+{cleaned}"
                
        return None

    def clean_email(self, email):
        """Clean and validate email address"""
        if not email:
            return None
            
        # Remove surrounding whitespace and lowercase
        email = email.strip().lower()
        
        # Basic validation
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return email
            
        return None

    def get_base_url(self, url):
        """Extract base URL from given URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def find_contact_pages(self, soup, base_url):
        """Find potential contact page URLs"""
        contact_urls = set(base_url)
        
        # Check all links
        for link in self.soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            text = link.get_text().lower()
            
            # Check if link text or URL contains contact-related words
            if any(word in text or word in href for word in self.contact_page_patterns):
                full_url = urljoin(base_url, href)
                contact_urls.add(full_url)
                
        return list(contact_urls)

    def extract_from_text(self, text):
        """Extract contact information from text content"""
        phones = set()
        emails = set()
        
        # Find phones
        for pattern in self.phone_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phone = self.clean_phone(match.group(0))
                if phone:
                    phones.add(phone)
                    
        # Find emails
        for pattern in self.email_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                email = self.clean_email(match.group(0))
                if email:
                    emails.add(email)
                    
        return list(phones), list(emails)
    

    def scrape_url(self, url):
        """Scrape a single URL for contact information"""
        try:
            # soup = self.fetch_and_parse_url(url)
            if not self.soup:
                return [], []
            
            # Remove script and style elements
            for element in self.soup(['script', 'style']):
                element.decompose()

            # Get text content
            text_content = self.soup.get_text()
            
            # Extract contact info from main page
            phones, emails = self.extract_from_text(text_content)
            
            # If no contact info found, try checking HTML attributes
            if not emails:
                for tag in self.soup.find_all(['a', 'input', 'meta']):
                    for attr in ['href', 'value', 'content']:
                        value = tag.get(attr, '').lower()
                        if '@' in value:
                            email = re.search(self.email_patterns[0], value)
                            if email:
                                cleaned_email = self.clean_email(email.group(0))
                                if cleaned_email:
                                    emails.append(cleaned_email)

            # If still no info found, try looking for contact pages
            if not phones and not emails:
                base_url = self.get_base_url(url)
                contact_urls = self.find_contact_pages(self.soup, base_url)
                
                for contact_url in contact_urls:
                    try:
                        response = requests.get(contact_url, headers=self.headers, timeout=10)
                        response.raise_for_status()
                        
                        contact_soup = BeautifulSoup(response.text, 'html.parser')
                        for element in contact_soup(['script', 'style']):
                            element.decompose()
                            
                        contact_text = contact_soup.get_text()
                        new_phones, new_emails = self.extract_from_text(contact_text)
                        
                        phones.extend(new_phones)
                        emails.extend(new_emails)
                        
                    except Exception as e:
                        self.logger.warning(f"Error scraping contact page {contact_url}: {str(e)}")

            # Remove duplicates while preserving order
            phones = list(dict.fromkeys(phones))
            emails = list(dict.fromkeys(emails))
            
            return phones, emails

        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return [], []

    def scrape_batch(self, urls):
        """Scrape multiple URLs for contact information"""
        results = []
        for url in urls:
            time.sleep(2)  # Be respectful to servers
            phones, emails = self.scrape_url(url)
            results.append({
                'url': url,
                'phones': phones,
                'emails': emails
            })
        return results
    
    # def fetch_webpage(self, url: str) -> Optional[str]:
    #     """
    #     Fetch the HTML content of a webpage
        
    #     :param url: URL of the webpage to scrape
    #     :return: HTML content or None if fetch fails
    #     """
    #     try:
    #         headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #         }
    #         response = requests.get(url, headers=headers, timeout=10)
    #         response.raise_for_status()
    #         return response.text
    #     except requests.RequestException as e:
    #         # print(f"Error fetching webpage: {e}")
    #         return None

    def detect_pms(self, html_content: str) -> List[str]:
        """
        Detect PMS systems in the HTML content using multiple strategies
        
        :param html_content: HTML content to search
        :return: List of detected PMS systems
        """
        detected_pms = []

        # Strategy 1: Direct text matching
        for pattern in self.pms_patterns:
            matches = pattern.findall(html_content)
            detected_pms.extend(matches)

        # Strategy 2: Search in HTML comments
        comment_matches = re.findall(r'<!--.*?-->', html_content, re.DOTALL)
        for comment in comment_matches:
            for pattern in self.pms_patterns:
                matches = pattern.findall(comment)
                detected_pms.extend(matches)

        # Strategy 3: Search in script tags
        script_matches = re.findall(r'<script[^>]*>(.*?)</script>', html_content, re.DOTALL)
        for script in script_matches:
            for pattern in self.pms_patterns:
                matches = pattern.findall(script)
                detected_pms.extend(matches)

        # Strategy 4: BeautifulSoup text extraction
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Search in specific meta tags
            meta_tags = [
                soup.find('meta', attrs={'name': 'generator'}),
                soup.find('meta', attrs={'name': 'application-name'}),
            ]
            for tag in meta_tags:
                if tag and tag.get('content'):
                    for pattern in self.pms_patterns:
                        matches = pattern.findall(tag['content'])
                        detected_pms.extend(matches)

            # Search in div and span tags with specific classes/ids
            suspicious_classes = ['powered-by', 'platform', 'booking-system']
            for cls in suspicious_classes:
                elements = soup.find_all(['div', 'span'], class_=re.compile(cls, re.IGNORECASE))
                for elem in elements:
                    for pattern in self.pms_patterns:
                        matches = pattern.findall(elem.get_text())
                        detected_pms.extend(matches)
        except Exception as e:
            # print(f"BeautifulSoup parsing error: {e}")
            pass

        # Remove duplicates and return unique detections
        return list(set(detected_pms))

    def analyze_website(self, url: str) -> List[str]:
        """
        Comprehensive PMS detection for a given URL
        
        :param url: URL to analyze
        :return: List of detected PMS systems
        """
        # soup = self.fetch_and_parse_url(url)
        if not self.soup:
            return []

        html_content =self.soup.prettify()

        return self.detect_pms(html_content)
    
    def detect_payment_gateway(self, html_content: str) -> List[dict]:
        """
        Detect payment gateways in the HTML content using multiple strategies
        """
        detected_gateways = []

        # Strategy 1: Search in HTML source
        for gateway, identifiers in self.payment_gateways.items():
            for identifier in identifiers:
                pattern = re.compile(rf'{identifier}', re.IGNORECASE)
                if pattern.search(html_content):
                    detected_gateways.append({
                        'gateway': gateway,
                        'identifier': identifier,
                        'location': 'HTML source'
                    })

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Strategy 2: Search in script tags
            script_tags = soup.find_all('script', {'src': True})
            for script in script_tags:
                src = script.get('src', '')
                for gateway, identifiers in self.payment_gateways.items():
                    for identifier in identifiers:
                        if identifier.lower() in src.lower():
                            detected_gateways.append({
                                'gateway': gateway,
                                'identifier': identifier,
                                'location': 'Script source'
                            })

            # Strategy 3: Search in form actions
            forms = soup.find_all('form')
            for form in forms:
                action = form.get('action', '')
                for gateway, identifiers in self.payment_gateways.items():
                    for identifier in identifiers:
                        if identifier.lower() in action.lower():
                            detected_gateways.append({
                                'gateway': gateway,
                                'identifier': identifier,
                                'location': 'Form action'
                            })

            # Strategy 4: Search in iframes
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                src = iframe.get('src', '')
                for gateway, identifiers in self.payment_gateways.items():
                    for identifier in identifiers:
                        if identifier.lower() in src.lower():
                            detected_gateways.append({
                                'gateway': gateway,
                                'identifier': identifier,
                                'location': 'iFrame source'
                            })

        except Exception as e:
            # print(f"Error parsing HTML: {e}")
            pass

        # Remove duplicates while preserving detection details
        unique_gateways = []
        seen = set()
        for gateway in detected_gateways:
            key = (gateway['gateway'], gateway['identifier'], gateway['location'])
            if key not in seen:
                unique_gateways.append(gateway)
                seen.add(key)

        return unique_gateways

    def analyze_checkout_flow(self, base_url: str) -> List[dict]:
        """
        Analyze the checkout flow for payment gateway detection
        """
        detected_gateways = []
        
        # Common checkout URL patterns
        checkout_patterns = [
            '/checkout',
            '/payment',
            '/book',
            '/reserve',
            '/booking',
            '/cart'
        ]
        
        # Check base URL first
        if self.soup:
            html_content = self.soup.prettify()
            detected_gateways.extend(self.detect_payment_gateway(html_content))
        
        # Check potential checkout pages
        for pattern in checkout_patterns:
            checkout_url = f"{base_url.rstrip('/')}{pattern}"
            html_content = self.fetch_and_parse_url(checkout_url)
            if html_content:
                html_content = html_content.prettify()
                detected_gateways.extend(self.detect_payment_gateway(html_content))
        
        return detected_gateways
    
    def format_number(self, phone):
        # Convert to string and remove any non-digit characters
        if pd.isna(phone):
            return phone
        
        phone_str = str(phone)
        digits = ''.join(filter(str.isdigit, phone_str))
        
        # Format based on number of digits
        if len(digits) == 10:
            return '+1' + digits
        elif len(digits) > 10:
            return '+' + digits
        else:
            return '+' + digits
    
    def full_web_scrapping(self, urls, progress_bar):

        """Scrape multiple URLs for contact information"""
        enriched_df = []
        i = 0
        for url_0 in urls:
            progress_bar.progress((i + 1) / len(urls))
            time.sleep(2)
            # Contact validation web scrapping
            if not url_0.startswith(('http://', 'https://')):
                url = 'https://' + url_0
            else:
                url = url_0
            
            self.soup = ''
            self.soup = self.fetch_and_parse_url(url)
            phones, emails = self.scrape_url(url)

            alternative_phone_number_1 = ''
            alternative_phone_number_2 = ''
            if len(phones) > 0:
                alternative_phone_number_1 = self.format_number(phones[0])
                if len(phones) > 1:
                    alternative_phone_number_2 = self.format_number(phones[1])
                                    
            
            alternative_email_1 = ''
            alternative_email_2 = ''
            if len(emails) > 0:
                alternative_email_1 = emails[0]
                if len(emails) > 1:
                    alternative_email_2 = emails[1]
                

            # PMS detection
            pms_systems = self.analyze_website(url)
            if len(pms_systems) > 0:
                pms = pms_systems[0]
            else:
                pms = ''

            # Gateway detection
            gateways = self.analyze_checkout_flow(url)
            if len(gateways) > 0:
                gt = gateways[0]
            else:
                gt = ''

            enriched_df.append({
                'url': url_0,
                'alternative phone_number_1': alternative_phone_number_1,
                'alternative phone_number_2': alternative_phone_number_2,
                'alternative email_1': alternative_email_1,
                'alternative email_2': alternative_email_2,
                'validated_phone_numbers': ' - '.join(phones),
                'validated_emails': ' - '.join(emails),
                'Detected PMS': pms,
                # 'Detected Gateway': gt
                })
            
        i += 1

        return pd.DataFrame(enriched_df) 
    
def enrich_full_data(df_companies, url_column: str, company_column: str, progress_bar):

    for index, row in df_companies.iterrows():
        if pd.isna(row[url_column]):
            company = row[company_column]
            # url_search = CompanySearch()
            urls = search_company(company)
            if urls:
                df_companies.at[index, url_column] = urls[0]
    
    df_companies.drop_duplicates(subset=[url_column], inplace=True)
    urls = df_companies[url_column].astype(str)
    detector = DataEnrichmentSystem()
    enriched_df = detector.full_web_scrapping(urls, progress_bar)
    return enriched_df
    # enriched_df.to_csv(path_output, index = False)