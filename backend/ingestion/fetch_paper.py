import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote

def fetch_arxiv_papers(query, max_results=5, output_dir="raw_data"):
    
    os.makedirs(output_dir, exist_ok=True)

    
    encoded_query = quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"

    print(f"Fetching metadata for '{query}' from arXiv API...")
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch data: HTTP {response.status_code}")
        return

    # 3. Parse the XML response
    root = ET.fromstring(response.content)
    # arXiv API uses the Atom XML namespace
    namespace = {'atom': 'http://www.w3.org/2005/Atom'} 

    # Find all 'entry' tags (each represents a paper)
    entries = root.findall('atom:entry', namespace)

    if not entries:
        print("No papers found for that query.")
        return

    # 4. Loop through the results and download the PDFs
    for entry in entries:
        # Get the title and strip weird characters/newlines for a safe filename
        title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
        safe_title = "".join([c for c in title if c.isalnum() or c.isspace()]).rstrip()
        # Truncate title so filenames don't get too long
        safe_title = safe_title[:60] 

        # Find the specific link tag that contains the PDF URL
        pdf_link = None
        for link in entry.findall('atom:link', namespace):
            if link.attrib.get('title') == 'pdf':
                pdf_link = link.attrib.get('href')
                # arXiv API returns links like 'http://arxiv.org/pdf/1234.5678v1'
                # We append .pdf to the URL just to be safe
                if not pdf_link.endswith('.pdf'):
                    pdf_link += '.pdf'
                break

        if pdf_link:
            filepath = os.path.join(output_dir, f"{safe_title}.pdf")
            print(f"Downloading: {safe_title}... ")
            
            # Download the actual PDF file
            pdf_response = requests.get(pdf_link)
            if pdf_response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                print(f" -> Saved to {filepath}")
            else:
                print(" -> Failed to download PDF.")
        else:
            print(f"No PDF link found for: {title}")

    print("\nBatch download complete!")

if __name__ == "__main__":
    # --- Configuration ---
    SEARCH_TOPIC = "quantum computing" # Change your keyword here
    PAPER_COUNT = 3                    # Number of papers to download
    TARGET_FOLDER = "raw_data"         # The directory to populate
    
    fetch_arxiv_papers(query=SEARCH_TOPIC, max_results=PAPER_COUNT, output_dir=TARGET_FOLDER)