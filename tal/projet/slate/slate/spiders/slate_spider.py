#!/usr/bin/env python
#-*- coding: utf-8 -*-
# scrapy runspider slate_spider.py -o info.json
import urlparse
import scrapy
import json as json

from slate.items import SlateItem

from HTMLParser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class SlateSpider(scrapy.Spider):
    name = "slate"
    start_urls = ["http://www.slate.fr"]

    def parse(self, response):
        
        for h in  response.css("article > a::attr('href')"):
            
            h = urlparse.urljoin(response.url, h.extract())
            

            yield scrapy.Request(h, callback=self.parsePage)
            
    
    def parsePage(self, response):
    	
        for sel in response.css('body'):
            item = SlateItem()
            item['titre'] = u'{}'.format(sel.css('h1::text').extract()[0])
            item['sousTitre'] = u'{}'.format(sel.css('.hat::text').extract()[0])    
            
            item['content'] = u"";

            for p in sel.css('.main_content > p'):

                item['content']+=strip_tags(p.extract())
            
            yield item
                
