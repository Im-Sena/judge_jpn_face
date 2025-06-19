from icrawler.builtin import GoogleImageCrawler
import datetime

n_total_images = 100
n_per_crawl = 100

delta = datetime.timedelta(days=720)
end_day = datetime.datetime(2023, 1, 29)

def datetime2tuple(date):
    return (date.year, date.month, date.day)

keywords = ['japanese actor','korean actor','chinese actor']

for keyword in keywords:
    for i in range(int(n_total_images / n_per_crawl)):
        start_day = end_day - delta
        #ディレクトリを国籍ごとに変える
        google_crawler = GoogleImageCrawler(downloader_threads=4, storage={'root_dir': './image ' + keyword.split()[0]})
        google_crawler.crawl(keyword=keyword, filters={'date': (datetime2tuple(start_day), datetime2tuple(end_day))}, file_idx_offset=i*n_per_crawl, max_num=n_per_crawl)
        end_day = start_day - datetime.timedelta(days=1)