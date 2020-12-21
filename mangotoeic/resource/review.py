from typing import List
from flask import request
from flask_restful import Resource, reqparse 
from mangotoeic.ext.db import engine
from flask import jsonify
import keras
 
from mangotoeic.ext.db import db, openSession
from sqlalchemy import func
 
import json



#############################3
###############################
###############################

###############################    model ###############################
###############################
###############################

import pandas as pd 
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from konlpy.tag import Okt

from mangotoeic.utils.file_helper import FileReader
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.dirname(os.path.abspath(__file__))
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
 
# from konlpy.tag import Kkma 시간 오래걸려

from sklearn.model_selection import train_test_split
 
class Prepro():
    def __init__(self):
     
        self.reader = FileReader()
        self.okt = Okt()
        self.df = self.get_data() 
        self.stopwords_list = []   

    def hook_process(self): 
        df = self.df
        df = Prepro.drop_col(df, col = 'label')
        self.stopword_list = self.get_stopwords()

        X = df['review'] 
        y = df['star'] 
        
        # 형태소별로 나눠줌
        word_tokens = Prepro.tokenize(data=X, stopword = self.stopword_list)
        print(word_tokens[:1])
        # 쓰이는 단어의 수와 인코딩해줄 단어 사전의 사이즈 추출
        vocabs = Prepro.vocab_size(tokenlist = word_tokens)
    
        # 정수 인코딩. 단어를 정수로 만들어주고 문장을 정수 벡터로 만들어줌
        encodedlist = Prepro.encoding(vocabs, tokenlist = word_tokens)
        print(encodedlist[:1])
        # 리뷰의 평균 단어 수
        length = Prepro.graph_review_length_frequency(encodedlist)

        # 이제 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 패딩 작업
        padded = Prepro.zeropadding(encodedlist, length = length)
        print(padded[:1])

        X = padded
        # 별점은 원핫 인코딩처리.
        y = Prepro.one_hot_encoding(y)
        X_train, X_test, y_train, y_test = Prepro.split(X,y)
         
        Prepro.accuracy_by_keras_LSTM(X_train, X_test, y_train, y_test, vocab_size_for_embedding = vocabs)
        # Prepro.accuracy_by_keras_RNN(X_train, X_test, y_train, y_test, vocab_size_for_embedding = vocabs)

    
        
    def get_stopwords(self):
        reader = self.reader
        reader.context = os.path.join(basedir, 'data')
        reader.fname = '불용어.txt'
        file = reader.new_file()
        f= open(file,'r', encoding='utf8')
        stopwords = f.read()
        f.close()
        stopword_list = stopwords.split('\n')
        return stopword_list

    def get_data(self): 
        reader = self.reader
        reader.context = os.path.join(basedir, 'data')
        reader.fname = "앱리뷰csv파일2.csv"
        reader.new_file()
        review_data = reader.csv_to_dframe()

        """이메일 추가해주는 부분"""
        # reader.context = os.path.join(basedir, 'data')
        # reader.fname = "100-contacts.csv"
        # reader.new_file()
        # q = reader.csv_to_dframe()
        # review_data.email[10409:12409] = q.email[3:2003]
        # review_data.to_csv('앱리뷰csv파일2.csv')

        return review_data
        # .iloc[10409:12409,:] 웹에 표기할 예제

    @staticmethod
    def tokenize(data,stopword):
        okt = Okt()
        wordtoken_list = []
        for line in data:
            onereview=[]
            word_tokens = okt.morphs(line)
            for word in word_tokens:
                if word not in stopword:
                    onereview.append(word)
            wordtoken_list.append(onereview) 
        return wordtoken_list

    @staticmethod
    def vocab_size (tokenlist):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokenlist)
        vocabs = len(tokenizer.word_index)
        print(f'총 단어 수 : {vocabs}')
        freq = 6
        rare_word_count = 0   
        total_vocab_freq = 0  
        rare_freq = 0  
        for word, count in tokenizer.word_counts.items():
            total_vocab_freq += count
            if(count < freq):
                rare_word_count += 1
                rare_freq += count
        print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(freq - 1, rare_word_count))
        print("희귀 단어 비율:", (rare_word_count / vocabs)*100)
        print("희귀 단어 등장 빈도 :", (rare_freq / total_vocab_freq)*100)
        vocabs -= rare_word_count
        print('단어 집합 최대 크기 :', vocabs)
        return vocabs

    @staticmethod
    def encoding(vocabsize, tokenlist):
        tokenizer = Tokenizer(vocabsize)
        tokenizer.fit_on_texts(tokenlist)
        return tokenizer.texts_to_sequences(tokenlist)

    @staticmethod
    def graph_review_length_frequency(X):
        print('리뷰 최대 길이 : {}'.format(max(len(l) for l in X))) 
        print('리뷰 평균 길이 : {}'.format(sum(map(len, X)) / len(X))) 
        print('95%의 리뷰의 길이는 {}개 이하'.format(pd.Series(map(len, X)).quantile(.95)))
        print('90%의 리뷰의 길이는 {}개 이하'.format(pd.Series(map(len, X)).quantile(.90)))
        print('85%의 리뷰의 길이는 {}개 이하'.format(pd.Series(map(len, X)).quantile(.85)))
        print('80%의 리뷰의 길이는 {}개 이하'.format(pd.Series(map(len, X)).quantile(.80)))
        plt.hist([len(s) for s in X], bins=50) 
        plt.xlabel('Review Length') 
        plt.ylabel('Number of Data') 
        plt.title('Review Count by Length')
        plt.show()
        length = int(round(pd.Series(map(len, X)).quantile(.90)))
        return length

    @staticmethod
    def zeropadding(encodedlist, length):
        padded = pad_sequences(encodedlist, padding = 'post', maxlen = length)
        return padded
    
    @staticmethod
    def one_hot_encoding(col):
        return np_utils.to_categorical(col)

    @staticmethod 
    def split(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def drop_col(df, col):
        return df.drop([col], axis = 1)

    @staticmethod
    def accuracy_by_keras_LSTM(X_train,X_test,y_train,y_test,vocab_size_for_embedding):
        seq = Sequential()
        seq.add(Embedding(vocab_size_for_embedding,50))
        seq.add(LSTM(18))
        seq.add(keras.layers.Dropout(.2))
        seq.add(Dense(5, activation='softmax'))

        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        checkpoint = ModelCheckpoint('review_star_model_lstm_11.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        run = seq.fit(X_train, y_train, callbacks=[earlystopping, checkpoint], epochs=10, batch_size=10, validation_split=0.2)

        plt.plot(run.history['loss'], label = 'train')
        plt.plot(run.history['val_loss'], label = 'test')
        plt.title('Model Train vs Validation Loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        seq.save("review_star_model_lstm_22")
        print('테스트 정확도 : {:.2f}%'.format(seq.evaluate(X_test,y_test)[1]*100))

        return seq.evaluate(X_test,y_test)

    @staticmethod
    def accuracy_by_keras_RNN(X_train,X_test,y_train,y_test,vocab_size_for_embedding):
        seq = Sequential()
        seq.add(Embedding(vocab_size_for_embedding+1,50))
        seq.add(SimpleRNN(10))
        seq.add(keras.layers.Dropout(.4))
        seq.add(Dense(5, activation='softmax'))
        
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        checkpoint = ModelCheckpoint('RNN_review_star_model_11.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        run = seq.fit(X_train, y_train, epochs=10, callbacks=[earlystopping, checkpoint], batch_size=10, validation_split=0.25, shuffle = True)
        plt.plot(run.history['loss'])
        plt.plot(run.history['val_loss'])
        plt.title('Model Train vs Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        seq.save("RNN_review_star_model_22.h5")
        print('테스트 정확도 : {:.2f}%'.format(seq.evaluate(X_test,y_test)[1]*100))

        return seq.evaluate(X_test,y_test)        
 




##########################################
##########################################
############## web crawling ##############
########################################## 
##########################################

 

import re
from bs4 import BeautifulSoup  
from selenium import webdriver
import time

class WebCrawler():
    def __init__(self):
        self.reviews = []
    
    def hook_process(self):
        # df = wc.webdata_toCsv(urls)
        # self.add_sentiment(df)
        self.get_data()
         

    def strip_emoji(self,text):
        RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        return RE_EMOJI.sub(r'', text)

    def cleanse(self,text):
        pattern = '[\r|\n]' # \r \n 제거
        text = re.sub(pattern,' ', text)
        RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        text =  RE_EMOJI.sub(r'', text) # 이모티콘 제거
        pattern = '([ㄱ-ㅎㅏ-ㅣ])+' # 한글 자음모음 제거
        text = re.sub(pattern,' ', text)
        pattern = '[^\w\s]' # 특수기호 제거
        text = re.sub(pattern, ' ', text)
        pattern = re.compile(r'\d+') # 숫자제거
        text= re.sub(pattern, ' ', text) 
        pattern = re.compile('[^ ㄱ-ㅣ가-힣]+') #영어 제거, 한글만 남기기
        text = re.sub(pattern, '', text)
        pattern = re.compile(r'\s+') # 띄어쓰기 여러개 붙어있을 시 제거
        text = re.sub(pattern,' ', text)
        return text

    def webdata_toCsv(self,urls):
        driver = webdriver.Chrome('mangotoeic/resource/data/chromedriver86.0424.exe')
        for i in range(len(urls)):
            url = urls[i]
            driver.get(url)
            driver.maximize_window()
            time.sleep(2)
            n=0
            nomorebutton=0
            while n<30 and nomorebutton < 5: # 3200개 뽑아줌
                for i in range(4):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    try:
                        driver.find_element_by_xpath("//span[@class='RveJvd snByac']").click()
                        n += 1
                        nomorebutton = 0
                    except Exception:
                        nomorebutton += 1   
            mysoup = BeautifulSoup(driver.page_source, 'html.parser')

            allreviews = mysoup.find_all('div', {'class':'d15Mdf bAhLNe'})
            
            for review in allreviews:
                score = review.find('div', {'role':'img'})['aria-label']
                star = score.split(' ')[3][0]
                comment = review.find('span', {'jsname':"bN97Pc"}).get_text()
                text = wc.cleanse(comment)
                if len(text) > 3:
                    self.reviews.append((text,star))
        driver.quit()    
        df = pd.DataFrame(self.reviews, columns = ['review','star'])
        return df

    def add_sentiment(self,df):
        df.loc[(df['star']>=4), 'label'] = 1
        df['label'] = df['label'].fillna(0)
        df.to_csv('앱리뷰csv파일.csv', index=False, encoding='utf-8-sig') 
        return df

    def get_data(self):
        reader = self.reader
        reader.context = basedir
        reader.fname = "앱리뷰csv파일2.csv"
        newfile=reader.new_file()
        review_data = reader.csv_to_dframe(newfile)
        return review_data.head(5)

urls = ['https://play.google.com/store/apps/details?id=com.taling&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.mo.kosaf&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.qualson.superfan&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.belugaedu.amgigorae&showAllReviews=true',
'https://play.google.com/store/apps/details?id=co.riiid.vida&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.hackers.app&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.pallo.passiontimerscoped&showAllReviews=true',
'https://play.google.com/store/apps/details?id=me.mycake&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.coden.android.ebs&showAllReviews=true',
'https://play.google.com/store/apps/details?id=kr.co.ebse.player&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.adrock.driverlicense300&showAllReviews=true',
'https://play.google.com/store/apps/details?id=net.tandem&showAllReviews=true',
'https://play.google.com/store/apps/details?id=kr.co.influential.youngkangapp&showAllReviews=true',
'https://play.google.com/store/apps/details?id=egovframework.tcpotal.mobile.lur&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.hackers.app.hackersmp3',
'https://play.google.com/store/apps/details?id=kr.go.hrd.app',
'https://play.google.com/store/apps/details?id=net.pedaling.class101&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.cjkoreaexpress&showAllReviews=true',
'https://play.google.com/store/apps/details?id=com.hackers.app.toeicvoca'
]

 
#####################################
#####################################
############DTO, SERVICE#############
#####################################
#####################################

class ReviewDto(db.Model):
    __tablename__ = "reviews"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: int = db.Column(db.Integer, primary_key=True, index=True)
    email : str = db.Column(db.String(500)) 
    review: str = db.Column(db.String(500))
    star: int = db.Column(db.Integer) 
  
    def __init__(self, id = None, email=None, review=None, star=None):
        self.id = id
        self.email = email
        self.review = review
        self.star = star 
    
    def __repr__(self):
        return f'Review(id=\'{self.id}\',email=\'{self.email}\',review=\'{self.review}\', star=\'{self.star}\',)'

    @property
    def json(self):
        return {
            'id' : self.id,
            'email' : self.email,
            'review' : self.review,
            'star' : self.star 
        }

class ReviewVo:
    id: int = 1
    email : str = ''
    review: str = ''
    star: int = 1
    

    
class ReviewService(object):

    @staticmethod
    def predict(input):
        wc = WebCrawler()
        model = Prepro() 
        cleansed_review = [wc.cleanse(input.review)]
        word_tokens = model.tokenize(data = cleansed_review, stopword = model.get_stopwords())
        encodedlist = model.encoding(vocabsize = 7132, tokenlist=word_tokens)
        reviewtext = model.zeropadding(encodedlist, 37)
        lstmmodel = keras.models.load_model('review_star_model_lstm_22')
        predictions = lstmmodel.predict(reviewtext)
        prob = np.max(predictions)*100
        prob = round(prob, 2)
        star = int(np.argmax(predictions))  
        return [prob,star]
        


# ==============================================================
# =====================                  =======================
# =====================    Dao    =======================
# =====================                  =======================
# ==============================================================



Session = openSession()
session = Session()

class ReviewDao(ReviewDto):
    
    @staticmethod
    def find_all(): 
        return session.query(ReviewDto).all()

    @classmethod 
    def find_by_email(cls,email): 
        print('FIND BY EMAIL ACTIVATED')
        return session.query(ReviewDto).filter(ReviewDto.email.like(f'%{email}%')).all() 


    @classmethod
    def find_by_review(cls,review): 
        print('FIND BY REVIEW ACTIVATED')
        return session.query(ReviewDto).filter(ReviewDto.review.like(f'%{review}%')).all()

    @staticmethod
    def save(review): 
        session.add(review)
        session.commit()
        
    @staticmethod
    def update(review): 
        session.add(review)
        session.commit()

    @staticmethod
    def delete(id): 
        print('123')
        session.query(ReviewDto).filter(ReviewDto.id == id).delete()
        session.commit()
    
    
    @staticmethod
    def count(): 
        return session.query(func.count(ReviewDto.id)).one()

    @staticmethod
    def insert_many():
        service = Prepro()
        df = service.get_data()
        print(df.head())
        session.bulk_insert_mappings(ReviewDto, df.to_dict(orient = 'records'))
        session.commit()
        session.close()
        print('done') 

     


# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================



class Review(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('email', type = str, required = True, help = 'This field should be email, cannot be left blank')
        parser.add_argument('review', type = str, required = True, help = 'This field should be review, cannot be left blank') 

        args = parser.parse_args()
        probstar = ReviewService.predict(args)
        print(f'예측 별점은 {probstar[0]}% 의 확률로 {probstar[1]}입니다')
        new_review = ReviewDto(email=args.email, review=args.review, star=probstar[1])
        print(new_review)
        try:
            ReviewDao.save(new_review) 
            return {'star': probstar[1], 'prob':probstar[0]}, 200
        except:
            return {'message' : ' an error occured while inserting review'}, 500 

    @staticmethod
    def get(review):
        try:
            review_searched = ReviewDao.find_by_review(review) 
            if review_searched: 
                lst= []
                for single_review_searched in review_searched:
                    srs = {
                            'id' : single_review_searched.id,
                            'email' : single_review_searched.email,
                            'review' : single_review_searched.review,
                            'star' : single_review_searched.star,
                            }
                    print(srs)
                    lst.append(srs) 
                return (lst), 200
        except Exception as e:
            return {'message': 'review_searched not found'}, 404 

    

     
 


class Review2(Resource):
    
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('email', type = str, required = True, help = 'This field should be email, cannot be left blank')
        parser.add_argument('review', type = str, required = True, help = 'This field should be review, cannot be left blank') 

        args = parser.parse_args()
        probstar = ReviewService.predict(args)
        print(f'예측 별점은 {probstar[0]}% 의 확률로 {probstar[1]}입니다')
        new_review = ReviewDto(email=args.email, review=args.review, star=probstar[1])
        print(new_review)
        try:
            ReviewDao.save(new_review) 
            return {'star': probstar[1], 'prob':probstar[0]}, 200
        except:
            return {'message' : ' an error occured while inserting review'}, 500 
 
    @staticmethod
    def delete(): 
        parser = reqparse.RequestParser()
        parser.add_argument('id', type = int, required = True, help = 'This field should be email, cannot be left blank')
        args = parser.parse_args()
        ReviewDao.delete(args.id)
        print('review remove complete!')
        return {'code' :0, 'message' : 'Success'}, 200
    
 

    @staticmethod
    def get(): 
        Session = openSession()
        session = Session()
        result = session.execute('select avg(star) from reviews;')
        data = result.first()
        result = round(data[0],2)  
        return str(result), 200


        
class Reviews(Resource): 

    @staticmethod
    def get():
        df = pd.read_sql_table('reviews', engine.connect()) 
        df.star = df.star + 1
        return json.loads(df.iloc[::-1].to_json(orient = 'records'))

    @staticmethod
    def post():
        rd = ReviewDao()
        rd.insert_many('reviews')



a= Prepro() 
a.hook_process()  

# b = ReviewService()
# wc = WebCrawler()
# model = Prepro()

 
# raw_review = wc.strip_emoji('게임 말고 그냥 단어랑 단어 뜻을 볼 수 있는 기능이 생기면 좋겠어요 게임 기능만 있으니 불편하네요')
# cleansed_review = [wc.cleanse(raw_review)]
# word_tokens = model.tokenize(data = cleansed_review, stopword = model.get_stopwords())
# encodedlist = model.encoding(vocabsize = 7132, tokenlist=word_tokens)
# reviewtext = model.zeropadding(encodedlist, 37) 
# lstmmodel = keras.models.load_model('review_star_model_lstm_22')
# predictions = lstmmodel.predict(reviewtext)
# prob = np.max(predictions)*100
# prob = round(prob, 2)
# star = int(np.argmax(predictions)) 

# print(prob,star)
 

# print(b.predict('게임 말고 그냥 단어랑 단어 뜻을 볼 수 있는 기능이 생기면 좋겠어요 게임 기능만 있으니 불편하네요'))
# print(b.predict('학원다닐 시간과 돈을 소비할 수 없는 저에겐 딱이네요 난이도도 그렇고 재미도있고 비용지불이 아깝지 않았습니다 굿이예요'))

