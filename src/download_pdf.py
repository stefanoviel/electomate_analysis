import requests
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time

def create_download_folder():
    """Create a folder for downloaded PDFs if it doesn't exist"""
    folder_name = "downloaded_pdfs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def is_pdf_url(url):
    """Check if URL directly points to a PDF"""
    return url.lower().endswith('.pdf')

def get_pdf_links_from_webpage(url):
    """Extract PDF links from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = []
        
        # Find all links
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.lower().endswith('.pdf'):
                # Convert relative URLs to absolute URLs
                if not href.startswith(('http://', 'https://')):
                    href = requests.compat.urljoin(url, href)
                pdf_links.append(href)
                
        return pdf_links
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return []

def download_pdf(url, folder):
    """Download a PDF file from a URL"""
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If filename is empty or doesn't end with .pdf, create a valid filename
        if not filename or not filename.lower().endswith('.pdf'):
            filename = f"document_{hash(url)}.pdf"
            
        filepath = os.path.join(folder, filename)
        
        # Download the file
        response = requests.get(url, timeout=10)
        
        # Check if content is actually PDF
        if 'application/pdf' in response.headers.get('content-type', '').lower():
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {filename}")
            return True
        else:
            print(f"Not a PDF file: {url}")
            return False
            
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create download folder
    download_folder = create_download_folder()
    
    # Read URLs from file
    with open('WahlprogrammeDEURLs.txt', 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    
    # Process each URL
    for url in urls:
        print(f"\nProcessing: {url}")
        
        # If URL directly points to PDF, download it
        if is_pdf_url(url):
            download_pdf(url, download_folder)
        else:
            # If it's a webpage, look for PDF links
            pdf_links = get_pdf_links_from_webpage(url)
            for pdf_url in pdf_links:
                print(f"Found PDF link: {pdf_url}")
                download_pdf(pdf_url, download_folder)
        
        # Add a small delay to be nice to servers
        time.sleep(1)

if __name__ == "__main__":
    main()