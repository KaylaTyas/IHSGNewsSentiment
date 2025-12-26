from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta, timezone
import requests
from bs4 import BeautifulSoup
import time
import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise Exception("Set DB_URL as env var first")
engine = create_engine(DB_URL, connect_args={"connect_timeout": 10})

MIN_ARTICLES = 15

def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(180)
    return driver

def format_detik_date(date_obj):
    return date_obj.strftime("%m/%d/%Y")

def scrape_article(url, kategori, fallback_date):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=14)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        judul_el = soup.find("h1", class_="detail__title")
        judul = judul_el.get_text(strip=True) if judul_el else ""
        tanggal_element = soup.find("div", class_="date")
        if tanggal_element:
            try:
                date_str = tanggal_element.get_text(strip=True).split(",")[1].strip()
                date_obj = datetime.strptime(date_str, "%d %b %Y")
                tanggal = date_obj.strftime("%Y-%m-%d")
            except:
                tanggal = fallback_date.strftime("%Y-%m-%d")
        else:
            tanggal = fallback_date.strftime("%Y-%m-%d")
        paragraphs = soup.select("div.detail__body-text.itp_bodycontent > p")
        konten = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        return {"tanggal": tanggal, "kategori": kategori, "judul": judul, "url": url, "konten": konten}
    except Exception as e:
        return None

def insert_raw_news(row):
    sql = text("""
        INSERT INTO raw_news (tanggal, kategori, judul, url, konten, scraped_at)
        VALUES (:tanggal, :kategori, :judul, :url, :konten, now())
        ON CONFLICT (url) DO NOTHING
        RETURNING id;""")
    with engine.begin() as conn:
        try:
            res = conn.execute(sql, {
                "tanggal": row["tanggal"],
                "kategori": row["kategori"],
                "judul": row["judul"],
                "url": row["url"],
                "konten": row["konten"]
            })
            return res.fetchone() is not None
        except:
            return False

def count_articles_for_date(date_str):
    sql = text("SELECT COUNT(*) FROM raw_news WHERE tanggal = :tanggal")
    with engine.connect() as conn:
        result = conn.execute(sql, {"tanggal": date_str}).scalar()
    return result or 0

def scrape_page(driver, url, kategori, current_date):
    try:
        driver.get(url)
        time.sleep(2)
        artikel_elements = driver.find_elements(By.CSS_SELECTOR, ".list-content__item")
        
        if not artikel_elements:
            return 0
        
        artikel_links = []
        for artikel in artikel_elements:
            try:
                link = artikel.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                artikel_links.append(link)
            except:
                continue
        
        inserted_count = 0
        for link in artikel_links:
            art = scrape_article(link, kategori, current_date)
            if art and art.get("konten"):
                if insert_raw_news(art):
                    inserted_count += 1
        
        return inserted_count
    except:
        return 0

def run_scrape(start_date, end_date):
    driver = create_driver()
    kategori = "ekonomi"
    current = start_date
    
    while current <= end_date:
        tanggal_str = format_detik_date(current)
        date_str_db = current.strftime("%Y-%m-%d")
        
        print(f"\n📅 {tanggal_str}", end=" ")
        
        existing_count = count_articles_for_date(date_str_db)
        
        if existing_count >= MIN_ARTICLES:
            print(f"✓ {existing_count} artikel")
            current += timedelta(days=1)
            continue
        
        page = 1
        total_new = 0
        while existing_count + total_new < MIN_ARTICLES:
            url = f"https://finance.detik.com/indeks?page={page}&date={tanggal_str}"
            new_inserted = scrape_page(driver, url, kategori, current)
            
            if new_inserted == 0:
                break  # Gak ada artikel di halaman ini
            
            total_new += new_inserted
            page += 1
            time.sleep(1)
        
        final_count = existing_count + total_new
        print(f"→ {final_count} artikel (+{total_new} baru)")
        
        current += timedelta(days=1)
    
    driver.quit()
    print("\n✅ Done")

def get_last_scraped_date():
    sql = text("SELECT MAX(tanggal) FROM raw_news;")
    with engine.connect() as conn:
        result = conn.execute(sql).scalar()
    return result

def get_scrape_range(lookback_days=3):
    today = datetime.now(timezone(timedelta(hours=7))).date()
    last_date = get_last_scraped_date()
    
    if last_date is None:
        start_date = today - timedelta(days=3)
    else:
        lookback_date = today - timedelta(days=lookback_days)
        start_date = max(lookback_date, last_date - timedelta(days=lookback_days))
    
    end_date = today
    
    if start_date > end_date:
        return None, None
    return start_date, end_date

if __name__ == "__main__":
    LOOKBACK_DAYS = 3 
    start_date, end_date = get_scrape_range(lookback_days=LOOKBACK_DAYS)
    
    if not start_date:
        print("✅ No new dates to scrape")
    else:
        print(f"🚀 Scraping: {start_date} → {end_date} (validasi {LOOKBACK_DAYS} hari)")
        run_scrape(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time())
        )