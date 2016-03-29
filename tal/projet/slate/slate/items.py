#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class SlateItem(scrapy.Item):
    # define the fields for your item here like:
    titre = scrapy.Field()
    sousTitre = scrapy.Field()
    content = scrapy.Field()

    pass
