from icrawler.builtin import BingImageCrawler #icrawlerのインポート

searchWord = '日本 芸能人 顔写真' #検索ワードの設定
crawler = BingImageCrawler(storage = {'root_dir' : './image'})
crawler.crawl(keyword = searchWord, max_num = 500)