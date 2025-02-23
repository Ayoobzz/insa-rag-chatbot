import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy import signals
from urllib.parse import urlparse
from twisted.internet import reactor
import json


class INSASpider(scrapy.Spider):
    name = "insa_spider"
    start_urls = ["https://www.insa-rennes.fr/informations-complementaires/plan-du-site.html"]

    custom_settings = {
        "DUPEFILTER_DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "DEPTH_LIMIT": 1,
        "ROBOTSTXT_OBEY": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_links = set()
        self.url_titles = {}

    def parse(self, response):
        title = response.css("title::text").get() or "No title"
        self.visited_links.add(response.url)

        yield {"url": response.url, "title": title}

        for link in response.css("a::attr(href)").getall():
            absolute_url = response.urljoin(link)
            if urlparse(absolute_url).netloc.endswith("insa-rennes.fr"):
                yield response.follow(absolute_url, self.parse)


def run_spider():
    visited_links = set()

    def handle_spider_closed(spider):
        nonlocal visited_links
        visited_links = spider.visited_links

    runner = CrawlerRunner()
    crawler = runner.create_crawler(INSASpider)
    crawler.signals.connect(handle_spider_closed, signal=signals.spider_closed)
    d = runner.crawl(crawler)
    d.addCallback(lambda _: reactor.stop())
    reactor.run()

    return visited_links



visited_links = run_spider()
print("Visited Links:", visited_links)
print(len(visited_links))
file_path = r"../data/raw/urls"

with open(file_path, "w") as f:
    json.dump(list(visited_links), f)


