import requests
from bs4 import BeautifulSoup
import urllib.parse


def scrape_incar_tags():
    """
    Scrapes all page titles (INCAR tags) from the 'Category:INCAR_tag' pages
    on the VASP Wiki, following 'next page' links until none remain.
    Returns a sorted list of unique tags with spaces replaced by underscores.
    """

    # Base URL and starting path
    base_url = "https://www.vasp.at"
    start_path = "/wiki/index.php?title=Category:INCAR_tag&pageuntil=DIMER+DIST#mw-pages"

    # Use a set to avoid duplicates
    all_tags = set()

    # Start from the given page
    next_page_url = base_url + start_path

    while next_page_url:
        # Fetch the page
        response = requests.get(next_page_url)
        if not response.ok:
            # If request fails, break out (or handle error as desired)
            break

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all tags in <div class="mw-category"> blocks
        category_divs = soup.find_all("div", class_="mw-category")
        for div in category_divs:
            links = div.find_all("a")
            for link in links:
                tag_text = link.get_text().strip()
                # Exclude navigation links like "(previous page)" or "(next page)"
                if tag_text and not tag_text.startswith("("):
                    # Replace spaces with underscores
                    underscore_tag = tag_text.replace(" ", "_")
                    all_tags.add(underscore_tag)

        # Look for the "next page" link, typically in <div id="mw-pages">
        next_page_link = None
        mw_pages_div = soup.find("div", id="mw-pages")
        if mw_pages_div:
            links = mw_pages_div.find_all("a")
            for link in links:
                text_lower = link.get_text().lower()
                # "next page" or "next 200" etc.
                if "next page" in text_lower or "next 200" in text_lower:
                    next_page_link = link.get("href")
                    break

        # Build the absolute URL for the next page or stop if none
        if next_page_link:
            next_page_url = urllib.parse.urljoin(base_url, next_page_link)
        else:
            next_page_url = None

    # Return a sorted list of tags
    return sorted(all_tags)


def main():
    # Run the scraping function
    tags = scrape_incar_tags()

    # Write all tags to valid_incar_tags.txt
    with open("valid_incar_tags.txt", "w") as f:
        for tag in tags:
            f.write(tag + "\n")

    # No print statements here, as requested.


if __name__ == "__main__":
    main()
