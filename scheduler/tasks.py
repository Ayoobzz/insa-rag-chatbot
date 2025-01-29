import redis
from apscheduler.schedulers.background import BackgroundScheduler


r = redis.Redis(host="localhost", port=6379, db=0)

def add_url_to_queue(url):
    r.lpush("insa_crawl_queue", url)

def get_next_url():
    return r.rpop("insa_crawl_queue")


scheduler = BackgroundScheduler()
@scheduler.scheduled_job("cron", hour=2)  # Run daily at 2 AM
def scheduled_scrape():
    scrape_insa_website()
    
scheduler.start()