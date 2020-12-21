import numpy as np
import nltk
import string
import re
import fasttext
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from mangotoeic.resource.vocablist import VocablistDto
from selenium import webdriver
import pandas as pd
from mangotoeic.resource.legacy import LegacyDto
import json
from mangotoeic.ext.db import db, openSession
from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from sqlalchemy import func
from mangotoeic.resource.chrome import sel_searching_data
from mangotoeic.resource.user import UserDto
from mangotoeic.resource.testresult import TestResultDto
from mangotoeic.resource.predictvocab import PredictVocabDto
import os
from mangotoeic.resource.vocabdict import VocabdictDto
from mangotoeic.resource.corpus import CorpusDto
from mangotoeic.resource.vocab import VocabDto
# from mangotoeic.resource.predictvocab import PredictVocabDto
import random
basedir= os.path.dirname(os.path.abspath(__file__))

class VocabRcdPro:
    def __init__(self):
        self.fpath = os.path.join(basedir,'./data/data.csv')
    
    def hook(self):
        df=self.fileread()
        print(df.head())
        return df

    def fileread(self):
        df = pd.read_csv(self.fpath,index_col=False,)
        df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(index ={0:'vocabId'})
        
        return df
    @staticmethod
    def lemmatized(corpus,id,qId):
        # ,id,answered_correctly,qId
        # vocab    user_id    answered_correctly  qid
        text = corpus.lower()
        regex=r'[!?\.\'\"\-\_\=\+\,\@\:\;\`\[\]\(\)\{\}\~\<\>\$0-9]'
        text = text.replace(regex,' ')
        vlist = text.split(' ')
        vset = set(vlist)
        vlist=list(vset)
        vlist =[contractions.fix(word) for word in text.split()]
        stop_words = nltk.corpus.stopwords.words('english')
        new_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'mr', 'ms', 'new', 'old', 'u', 'st', 'nd', 'rd', 'th']
        stop_words.extend(new_words)
        vlist=[word for word in vlist if word not in stop_words]
        vlist=nltk.tag.pos_tag(vlist)
        print(vlist)
        vlist=[(word,pos_tag) for word, pos_tag in vlist if pos_tag != 'NNP' and pos_tag != 'NNPS']
        vlist
        
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        vlist=[(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in vlist]
        wnl = WordNetLemmatizer()
        vlist= [wnl.lemmatize(word, tag) for word, tag in vlist]
        vlist= [item.replace(r'.','') for item in vlist ]
        vlist= [item.replace(r',','') for item in vlist ]
        vlist= [item.replace(r"'",'') for item in vlist ]
        vlist= [item.replace(r"0-9",'') for item in vlist]
        vlist= [item.replace(r"-",'') for item in vlist ]
        vlist= [item.replace(r"[0-9]",'') for item in vlist]
        vlist= [item.replace(r"0-9",'') for item in vlist]
        stop_words = nltk.corpus.stopwords.words('english')
        new_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'mr', 'ms', 'new', 'old', 'u', 'st', 'nd', 'rd', 'th']
        stop_words.extend(new_words)
        vlist=[word for word in vlist if (word not in stop_words) or (word!="" ) or (word==" ")]        
        print(vlist)
        df= pd.DataFrame(vlist, columns=['vocab'])
        df['user_id']=id
        df['qId']= qId
        print(df)
        return df

class VocabRcdDto(db.Model):
    
    __tablename__ = 'vocabrcd'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    vocabId: int = db.Column(db.Integer, primary_key=True, index=True)
    vocab: str = db.Column(db.String(50))
    user_id: int = db.Column(db.Integer,db.ForeignKey('users.user_id'))
    qId =db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    correctAvg : float = db.Column(db.Float)
    user_avg: float = db.Column(db.Float)

    @property
    def json(self):
        return {            
            'user_id' : self.user_id,
            'vocabId' : self.vocabId,
            'vocab': self.vocab,
            'correctAvg': self.correctAvg,
            'user_avg': self.user_avg
        }
    
class VocabRcdVo:
    user_id: int = 0
    vocabId: int = 0
    vocab: str = ''
    correctAvg: float = 0.0
class VocabRcdDao(VocabRcdDto):
    @staticmethod
    def hook2(id):

        VocabRcdDao.set_datas_from_user_id(id)
        
    @staticmethod
    def hook(id):

        VocabRcdDao.set_datas_from_user_id(id)
        dtos=VocabRcdDao.get_dtos_from_user_id(id)
        mylist=VocabRcdDao.rcd(dtos,id)
        return mylist
    @staticmethod
    def set_datas_from_user_id(id):
        testresult_dtos=TestResultDto.query.filter_by(user_id=id).all()
        for item in testresult_dtos:
            answered_correctly=item.answered_correctly
            qId =item.qId
            legacydto=LegacyDto.query.filter_by(qId= qId).first()
            answer=legacydto.answer
            question=legacydto.question
            regex =r'___'
            corpus =re.sub(regex,answer,question)
            print(corpus)
            df=VocabRcdPro.lemmatized(corpus,id,qId)
            q=VocabRcdDto.query.filter_by(user_id=id)
            df2= pd.read_sql(q.statement,q.session.bind)
            mylist=df2.vocab.to_list()
            print(mylist)
            print(~df.vocab.isin(mylist))
            df=df.loc[~df.vocab.isin(mylist)]
            print(df)
            VocabRcdDao.bulk(df)
            # break
        VocabRcdDao.get_average()
        VocabRcdDao.get_average2()
        
    @staticmethod
    def get_average():
        db.session.execute('update vocabrcd as t inner join (select user_id, avg(answered_correctly) as av from testresult group by user_id ) t1 on t.user_id = t1.user_id set t.user_avg= t1.av;')
        db.session.commit()
        db.session.close()

    @staticmethod
    def get_average2():
        db.session.execute('update vocabrcd as t inner join (select user_id,qid, avg(answered_correctly) as av from testresult group by user_id, qId ) t1 on t.user_id = t1.user_id and t.qId= t1.qId  set t.correctAvg= t1.av;')
        db.session.commit()
        db.session.close()
        
    @staticmethod
    def get_dtos_from_user_id(id):
        vocabrcddtos=db.session.query(VocabRcdDto).filter_by(user_id=id).all()
        return vocabrcddtos

    @staticmethod   
    def bulk(df):
        Session = openSession()
        session = Session()
        # print(df.head())
        session.bulk_insert_mappings(VocabRcdDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @staticmethod
    def rcd(dtos,id):
        users=UserDto.query.all()
        # print(users)
        minvalue= 100000
        minuser=None
        for user in users:
            if user.user_id>=16:
                continue
            rcddtos=VocabDto.query.filter_by(user_id=user.user_id).all()
            # print(rcddtos)
            mylist=[]
            for dto in dtos:
                mylist.append(dto.vocab)
            minis=dtos
            for mini in minis:
                mylist.append(mini.vocab)
            
            mylist = set(mylist)
            sum =0
            for rcditm in rcddtos:
                if rcditm.vocab in mylist:
                    # print("=="*100,rcditm)
                    minidto =VocabRcdDto.query.filter_by(user_id=id,vocab=rcditm.vocab).first()
                    # print(minidto)
                    value=abs(rcditm.correctAvg-minidto.correctAvg)
                    sum+=value
                elif not rcditm.vocab in mylist:
                    sum2=0
                    count=0
                    for rcditm2 in rcddtos:
                        sum2+=rcditm2.correctAvg
                        count+=1
                    avg_of_user_from_rcd= sum2/count
                    minidto =VocabRcdDto.query.filter_by(user_id=id).first()
                    # print(minidto)
                    value=abs(avg_of_user_from_rcd-minidto.user_avg)
                    sum+=value
            corrent_value=sum
            if corrent_value<minvalue:
                minvalue=corrent_value
                minuser=user.user_id
            
        print(minuser)
        print(minvalue)
        q=PredictVocabDto.query.filter_by(user_id=minuser)
        df= pd.read_sql(q.statement,q.session.bind)
        df_sorted_by_values=df.sort_values(by='correctAvg',ascending =False)
        print(df_sorted_by_values)
        p=df_sorted_by_values.iloc[int(len(df)/2)]
        print(p)
        median=p['correctAvg']
        mfdtos=q.all()
        mylist2=[]
        for mfdto in mfdtos:
            # mfdto중 가장 중간 오답률을 찾는다
            difference=median-mfdto.correctAvg
            if abs(difference)>0.01:
                continue
            if difference < 0: #맞출확률이 높다면
                x=random.randint(0,1)
                if x==0: 
                    continue
                if x==1:
                    pass
            mylist2.append(mfdto)
        
        mylist4=[]
        for item in mylist2:
            vocabdictdtos=VocabdictDto.query.filter_by(vocab=item.vocab).all()
            mylist3=[]
            mydict={}
            for vocabdictdto in vocabdictdtos:
                jsonobj=vocabdictdto.json
                mylist3.append(jsonobj['meaning'])
                print(jsonobj)
                vocab=jsonobj['vocab']
                mydict[vocab]= mylist3
            mylist4.append(mydict)
        print(mylist4)
        return mylist4

class VocabRcds(Resource):
    @staticmethod
    def get(id):
        data = VocabRcdDao.hook(id)
        return data, 200
class VocabBulk(Resource):
    @staticmethod
    def get(id):
        VocabRcdDao.hook2(id)
if __name__ == '__main__':
    # corpus="The assets of Marble Faun Publishing Company suffered last quarter when one of their main local distributors went out of business."
    # VocabRcdPro.lemmatized(corpus,1,0,12)
    VocabRcdDao.set_datas_from_user_id(16)
    


        