# -*- coding: utf-8 -*-
from icrawler.builtin import BingImageCrawler
import datetime

n_total_images = 500
n_per_crawl = 500

delta = datetime.timedelta(days=3000)
end_day = datetime.datetime(2025, 6, 29)

keywords = [
    'japanese actor', 'japanese actress', 'japanese celebrity',
    'korean actor', 'korean actress', 'korean celebrity',
    'chinese actor', 'chinese actress', 'chinese celebrity'
]

for keyword in keywords:
    for i in range(int(n_total_images / n_per_crawl)):
        dir_name = './image ' + keyword.split()[0]
        bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': dir_name})
        bing_crawler.crawl(
            keyword=keyword,
            file_idx_offset=i*n_per_crawl,
            max_num=n_per_crawl
        )