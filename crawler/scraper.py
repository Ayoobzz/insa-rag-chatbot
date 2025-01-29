from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    return webdriver.Chrome(options=options)

def extract_text(url, driver):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    for element in soup(["nav", "footer", "script", "style"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)