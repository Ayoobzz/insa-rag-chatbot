import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time


def get_main(driver, url):
    driver.get(url)
    time.sleep(3)  # Wait for JS to load
    html = driver.page_source  # Get the fully rendered HTML
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else "No title"
    main_content = soup.find("main")
    result = main_content.get_text(strip=True) if main_content else "Content not found"
    return result, title


def scrape_list(urls):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        results = {}
        for url in urls:
            res, title = get_main(driver, url)
            results[url] = [res, title]
    finally:
        driver.quit()

    return results


file_path = r"../data/raw/urls"
with open(file_path, "r") as f:
    my_list = json.load(f)
    results = scrape_list(my_list)
    with open(r"../data/processed/results.json", "w") as f:
        json.dump(results, f)