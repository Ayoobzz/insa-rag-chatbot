import scrapy
from scrapy.crawler import CrawlerProcess
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

TARGET_URLS = [
    "https://www.insa-rennes.fr/informations-complementaires/plan-du-site.html"
    ]

class INSASpider(scrapy.Spider):
    name = "insa_spider"
    start_urls = ["https://www.insa-rennes.fr"]

    def parse(self, response):
        # Extract links and filter by domain
        links = response.css("a::attr(href)").getall()
        for link in links:
            if "insa-rennes.fr" in link:
                yield response.follow(link, self.parse_page)

    def parse_page(self, response):
        # Extract text and metadata
        yield {
            "url": response.url,
            "text": response.css("body::text").getall(),
            "title": response.css("title::text").get()
        }

def login(driver):
    driver.get("https://intra.insa-rennes.fr/login")
    driver.find_element(By.ID, "username").send_keys("your_username")
    driver.find_element(By.ID, "password").send_keys("your_password")
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()


def extract_text(url, driver):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    # Remove navbars, footers, ads
    for element in soup(["nav", "footer", "script", "style"]):
        element.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text

# Run the spider


#
