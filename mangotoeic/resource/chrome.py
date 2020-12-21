from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
import os
import pickle
from time import sleep
import re
from collections import defaultdict
basedir =os.path.dirname(os.path.abspath(__file__))
print(basedir) 
def sel_searching_data(driver, item, vocabdict):
    
    bs4Sel=Bs4Sel()
    bs4Sel.sel_get_url(driver,'https://en.dict.naver.com/#/main')
    css='#ac_input'
    bs4Sel.sel_wait_by_css(driver,css)
    err=bs4Sel.sel_wait_by_css(driver,css)
    if not err:
        return vocabdict
    element=bs4Sel.sel_get_element_by_css(driver,css)
    err=bs4Sel.sel_wait_by_css(driver,css)
    if not err:
        return vocabdict
    text=item #
    bs4Sel.sel_input_text(driver,element,text)
    css='#searchPage_entry > div > div:nth-child(1) > div > a > strong'
    err=bs4Sel.sel_wait_by_css(driver,css)
    if not err:
        return vocabdict
    # print("done")
    element=bs4Sel.sel_get_element_by_css(driver, css)
    bs4Sel.sel_click(driver,element)
    css='em.part_speech'
    err=bs4Sel.sel_wait_by_css(driver, css)
    if not err:
        return vocabdict
    # proNs=bs4Sel.sel_get_elements_by_css(driver,css)
    # for proN in proNs:
    #     proN = proN.get_attribute('innerHTML')
    #     proN = proN.strip('\n\t')
        # print(proN)
    err=bs4Sel.sel_wait_by_css(driver, css)
    if not err:
        return vocabdict

    css='span.mean'
    meanings= bs4Sel.sel_get_elements_by_css(driver, css)
    css='#content > div.article._article.is-closed > div.section.section_mean._section_mean._data_index_1 > div > div.mean_tray > ul > li > div.mean_desc > span'
    nums=bs4Sel.sel_get_elements_by_css(driver,css)
    err=bs4Sel.sel_wait_by_css(driver, css)
    if not err:
        return vocabdict
    idx=0
    mydict={}
    mylist1= []
    for mean in meanings:
        mean = mean.get_attribute('innerHTML')
        mean = mean.strip()
        mean1=re.sub(r'(\<.*?\>)', '', mean)
        # numtxt = num.get_attribute('innerHTML')
        # numtxt = numtxt.strip()
        # if numtxt == '1.':
        #     proN = proNs[idx].get_attribute('innerHTML')
        #     idx+=1
        #     proN = proN.strip('\n\t')

            # print(proN)
        # print(numtxt+mean1)
        mylist1.append(mean1)
        # print(mylist1)
    # mylist2 = set(mylist1)
    
    vocabdict[item]=mylist1
    with open(os.path.join(basedir,"data/vocabdict6.pickle"),'wb') as f:
        pickle.dump(vocabdict,f)
    with open(os.path.join(basedir,"data/vocabdict6.pickle"),'rb') as f:
        data=pickle.load(f)
    print(data)
    return vocabdict
class Bs4Sel:
    def __init__(self):
        pass       
    @staticmethod
    def sel_get_element_by_css(driver,css): 
        
        element=driver.find_element_by_css_selector(css)
        
        return element
    @staticmethod
    def sel_get_elements_by_css(driver,css): 
     
        elements=driver.find_elements_by_css_selector(css)
        return elements
    @staticmethod
    def sel_get_url(driver,url):
      
        driver.get(url)
        
    @staticmethod
    def sel_wait_by_css(driver,css):
        try:
            sleep(3)
            delay =10
            WebDriverWait(driver,delay).until(expected_conditions.presence_of_element_located((By.CSS_SELECTOR, css)))
        except:
            return None
        return True
    @staticmethod
    def sel_input_text(driver,element,text):
     
        ActionChains(driver).move_to_element(element).click(element).send_keys(text, Keys.ENTER).perform()
    @staticmethod
    def sel_click(driver,element):
        ActionChains(driver).move_to_element(element).click(element).perform()
    # def sel_find_css()
if __name__ == '__main__':
    driver =webdriver.Chrome(os.path.join(basedir,'chromedriver'))
    # driver =webdriver.PhantomJS('/usr/local/bin/phantomjs')
    import pickle
    with open(os.path.join(basedir,"data/data.pickle"),'rb') as f:
        mylist=pickle.load(f)
        # print(mylist)
    import pandas as pd
    import numpy as np
    with open(os.path.join(basedir,'data/vocabdict.pickle'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(basedir,'data/vocabdict2.pickle'), 'rb') as f:
        data3 = pickle.load(f)
    with open(os.path.join(basedir,'data/vocabdict3.pickle'), 'rb') as f:
        data4 = pickle.load(f)
    with open(os.path.join(basedir,'data/vocabdict4.pickle'), 'rb') as f:
        data5 = pickle.load(f)
    with open(os.path.join(basedir,'data/vocabdict5.pickle'), 'rb') as f:
        data6 = pickle.load(f)
    with open(os.path.join(basedir,'data/data.pickle'), 'rb') as f:
        data2 = pickle.load(f)
    vocablist = list(data.keys())
    vocabset=set(vocablist)
    vocablist2 = list(data3.keys())
    vocabset2 = set(vocablist2)
    vocablist3 = list(data4.keys())
    vocabset3 = set(vocablist3)
    vocablist4 = list(data5.keys())
    vocabset4 = set(vocablist4)
    vocablist5 = list(data6.keys())
    vocabset5 = set(vocablist5)
    Todoset=data2-vocabset-vocabset2-vocabset3-vocabset4-vocabset5
    vocabdict={}
    mylist =list(Todoset)
    for item in mylist:
        # break
        vocabdict=sel_searching_data(driver, item, vocabdict)    
    with open(os.path.join(basedir,"data/vocabdict6.pickle"),'wb') as f:
        pickle.dump(vocabdict,f)
    with open(os.path.join(basedir,"data/vocabdict6.pickle"),'rb') as f:
        data=pickle.load(f)
    print(data)