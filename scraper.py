from bs4 import BeautifulSoup as soup  # HTML data structure
from urllib.request import urlopen as uReq  # Web client
import pandas as pd

# URl to web scrap from.
# in this example we web scrap graphics cards from Newegg.com

page_urls = [
    "https://cn.nytimes.com/asia-pacific/20190419/north-korea-economy-sanctions/dual/",
    "https://cn.nytimes.com/world/20171013/india-buddhist-muslim-marriage/dual/",
    "https://cn.nytimes.com/usa/20190422/chinese-immigrants-catholicism/dual/",
    "https://cn.nytimes.com/world/20190422/sri-lanka-bombings-what-we-know/dual/",
    "https://cn.nytimes.com/opinion/20190422/north-korea-russia-summit/dual/",
    "https://cn.nytimes.com/opinion/20190418/iphone-privacy/dual/",
    "https://cn.nytimes.com/opinion/20190419/chappatte-mueller-report-trump/dual/",
    "https://cn.nytimes.com/opinion/20190417/notre-dame-cathedral-fire/dual/",
    "https://cn.nytimes.com/opinion/20190411/c11qian-privacy/dual/",
    "https://cn.nytimes.com/opinion/20190409/trump-china-trade-deal/dual/",
    "https://cn.nytimes.com/opinion/20190408/xi-jinping-thought-app/dual/",
    "https://cn.nytimes.com/opinion/20190408/europe-china-trade/dual/",
    "https://cn.nytimes.com/opinion/20190404/apple-steve-jobs/dual/",
    "https://cn.nytimes.com/opinion/20190402/kellyanne-george-conway-marriage/dual/",
    "https://cn.nytimes.com/opinion/20190402/joe-biden-lucy-flores/dual/",
    "https://cn.nytimes.com/opinion/20190401/women-leadership-jacinda-ardern/dual/",
    "https://cn.nytimes.com/china/20170119/china-chief-justice-courts-zhou-qiang/dual/",
    "https://cn.nytimes.com/usa/20190419/white-house-mueller-report/dual/",
    "https://cn.nytimes.com/asia-pacific/20190419/north-korea-economy-sanctions/dual/",
    "https://cn.nytimes.com/usa/20190419/mueller-report-pdf-takeaways/dual/",
    "https://cn.nytimes.com/business/20190417/trump-china/dual/"]
table = pd.DataFrame()
english_col, chinese_col = [], []

# name the output file to write to local disk
out_filename = "chinese-english.csv"

for page_url in page_urls:
    # opens the connection and downloads html page from url
    uClient = uReq(page_url)

    # parses html into a soup data structure to traverse html
    # as if it were a json data type.
    page_soup = soup(uClient.read(), "html.parser")

    # finds each paragraph containing the article text
    containers = page_soup.findAll("div", {"class": "article-paragraph"})

    container_len = len(containers)
    assert container_len % 2 == 0
    even = [i for i in range(0, container_len, 2)]
    odd = [i+1 for i in even]

    for e, o in zip(even, odd):
        english_col.append(containers[e].text)
        chinese_col.append(containers[o].text)

uClient.close()

table['english'] = english_col
table['chinese'] = chinese_col
table.to_csv(out_filename, index=False)
