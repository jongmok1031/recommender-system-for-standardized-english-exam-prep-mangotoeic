from mangotoeic.resource.vocablist import VocablistDto
from selenium import webdriver
import pandas as pd
import json
from mangotoeic.ext.db import db, openSession
from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from sqlalchemy import func
from mangotoeic.resource.chrome import sel_searching_data
import pickle
import os
basedir= os.path.dirname(os.path.abspath(__file__))

class PredictVocabPro:
    def __init__(self):
        self.fpath = os.path.join(basedir,'./data/vocabmf.csv')
        self.fpath2 = os.path.join(basedir, './data/idx2vocab.pickle')
        self.fpath3 = os.path.join(basedir, './data/vocabid2userid.pickle')
    
    def hook(self):
        df=self.fileread()
        print(df.head())
        return df

    def fileread(self):
        df = pd.read_csv(self.fpath)
        with open(self.fpath2, 'rb') as f:
            data = pickle.load(f)
        with open(self.fpath3, 'rb') as f:
            data2 = pickle.load(f)
        print(data2)
        df = df.drop('Unnamed: 0', axis=1)
        df = df.unstack()
        df2=df.to_frame()
        # print(df2)
        df2=df2.rename(columns={ 0: "correctAvg"})
        df2.index.names =['user_id','vocab']
        df2 = df2.reset_index()
        df2 = df2.replace({"vocab":data})
        df3 = df2.replace({"user_id":data2})
        print(df3)
        return df3

class PredictVocabDto(db.Model):
    
    __tablename__ = 'predictvocab'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    vocabId: int = db.Column(db.Integer, primary_key=True, index=True)
    vocab: str = db.Column(db.String(50),db.ForeignKey('vocablist.vocab'))
    user_id: int = db.Column(db.Integer)
    correctAvg : float = db.Column(db.Float)

    def __repr__(self):
        return f' user_id={self.user_id}, vocabId={self.vocabId}, vocab={self.vocab},correctAvg={self.correctAvg}'

    @property
    def json(self):
        return {            
            'user_id' : self.user_id,
            'vocabId' : self.vocabId,
            'vocab': self.vocab,
            'correctAvg': self.correctAvg
        }

class PredictVocabDao(PredictVocabDto):
    @staticmethod   
    def bulk():
        service = PredictVocabPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        session.bulk_insert_mappings(PredictVocabDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(PredictVocabDto.vocab)).one()
    
if __name__ == "__main__":
    recommend = PredictVocabDao()
    recommend.bulk()