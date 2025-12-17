
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = "shl_products.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_list_pages():
    products = []
    start = 0
    type_param = 1 # Individual Test Solutions
    
    print("Step 1: Scraping Catalog List...")
    
    while True:
        url = f"{BASE_URL}?start={start}&type={type_param}"
        print(f"  Fetching {url}...")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"  Failed: {resp.status_code}")
                break
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            rows = soup.find_all('tr', attrs={'data-entity-id': True})
            
            if not rows:
                print("  No more rows found.")
                break
                
            for row in rows:
                # 1. Title & URL
                title_td = row.find('td', class_='custom__table-heading__title')
                if not title_td: continue
                link = title_td.find('a')
                if not link: continue
                
                title = link.get_text(strip=True)
                href = link['href']
                if not href.startswith('http'):
                    href = "https://www.shl.com" + href
                
                # 2. Remote Support (2nd column, index 1)
                cells = row.find_all('td')
                remote_support = "No"
                if len(cells) > 1:
                    circle = cells[1].find('span', class_='catalogue__circle -yes')
                    if circle:
                        remote_support = "Yes"

                # 3. Adaptive Support (3rd column, index 2)
                adaptive_support = "No"
                if len(cells) > 2:
                    circle = cells[2].find('span', class_='catalogue__circle -yes')
                    if circle:
                        adaptive_support = "Yes"
                
                # 4. Test Types
                test_types = []
                keys_td = row.find('td', class_='product-catalogue__keys')
                if keys_td:
                    keys = keys_td.find_all('span', class_='product-catalogue__key')
                    test_types = [k.get_text(strip=True) for k in keys if k.get_text(strip=True)]
                
                # Check mapping for Types to Full Names
                # A=Ability, B=Behavior, C=Competency, P=Personality, K=Knowledge, S=Simulations
                type_map = {
                    "A": "Ability & Aptitude",
                    "B": "Biodata & Situational Judgement",
                    "C": "Competencies",
                    "D": "Development & 360",
                    "E": "Assessment Exercises",
                    "K": "Knowledge & Skills",
                    "P": "Personality & Behavior",
                    "S": "Simulations"
                }
                full_test_types = [type_map.get(t, t) for t in test_types]

                products.append({
                    "name": title,
                    "url": href,
                    "remote_support": remote_support,
                    "adaptive_support": adaptive_support,
                    "test_type": full_test_types,
                    "description": "", # To be filled
                    "duration": 0      # To be filled
                })
            
            start += 12
            # Safety break
            # if len(products) > 20: break 
            
        except Exception as e:
            print(f"  Error on list page: {e}")
            break
            
    print(f"  Collected {len(products)} products initial info.")
    return products

def fetch_product_details(product):
    url = product['url']
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Duration
            # H4 sibling logic or text search
            duration = 0
            text_body = soup.get_text()
            
            # Try specific structure first
            # <h4>Assessment length</h4> <p>... 49 ...</p>
            h4_dur = soup.find('h4', string=re.compile(r'Assessment length|Time', re.I))
            if h4_dur:
                p_next = h4_dur.find_next_sibling('p')
                if p_next:
                    dur_text = p_next.get_text(strip=True)
                    match = re.search(r'(\d+)', dur_text)
                    if match:
                        duration = int(match.group(1))
            
            if duration == 0:
                # Fallback to regex in full text
                match = re.search(r'(?:Time|Duration).*?(\d+)\s*(?:min|minute)', text_body, re.I)
                if match:
                    duration = int(match.group(1))
            
            product['duration'] = duration
            
            # Description
            # <h4>Description</h4> <p>...</p>
            desc = ""
            h4_desc = soup.find('h4', string=re.compile(r'Description', re.I))
            if h4_desc:
                p_next = h4_desc.find_next_sibling('p')
                if p_next:
                    desc = p_next.get_text(strip=True)
            
            if not desc:
                # Fallback to .product-catalogue-training-calendar__row typ p (first paragraph)
                container = soup.find('div', class_='product-catalogue-training-calendar__row typ')
                if container:
                    p_tag = container.find('p')
                    if p_tag:
                         desc = p_tag.get_text(strip=True)

            if not desc:
                # Fallback to meta description
                meta = soup.find('meta', attrs={'name': 'description'})
                if meta:
                    desc = meta.get('content', '')
            
            product['description'] = desc
            
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    
    return product

def main():
    products = scrape_list_pages()
    
    # Filter duplicates just in case
    # Remove Pre-packaged if they slipped in (though we used type=1)
    
    print("Step 2: Fetching details for all products (Threaded)...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_prod = {executor.submit(fetch_product_details, p): p for p in products}
        
        completed = 0
        forfuture = as_completed(future_to_prod)
        for future in forfuture:
            completed += 1
            if completed % 20 == 0:
                print(f"  Processed {completed}/{len(products)}...")
                
    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(products, f, indent=2)
        
    print(f"Done. Saved {len(products)} enriched products to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
