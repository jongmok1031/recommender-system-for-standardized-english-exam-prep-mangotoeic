from mangotoeic.resource.vocablist import VocablistDto
from mangotoeic.resource.vocabdict import VocabdictDto
from selenium import webdriver
import pandas as pd
import json
from mangotoeic.ext.db import db, openSession
from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
from sqlalchemy import func
from mangotoeic.resource.chrome import sel_searching_data
import os
basedir= os.path.dirname(os.path.abspath(__file__))

class VocabPro:
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

class VocabDto(db.Model):
    
    __tablename__ = 'vocab'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    vocabId: int = db.Column(db.Integer, primary_key=True, index=True)
    vocab: str = db.Column(db.String(50),db.ForeignKey('vocablist.vocab'))
    user_id: int = db.Column(db.Integer)
    # db.ForeignKey(MinitestDto.qId)
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
    
class VocabVo:
    user_id: int = 0
    vocabId: int = 0
    vocab: str = ''
    correctAvg: float = 0.0

class VocabDao(VocabDto):
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def find_by_id(cls, userid):
        p = VocabDto.query.filter_by(user_id = userid).all()
        vocablist = []
        # print(p)
        for item in p:
            vocab=item.vocab
            vocabdict={}
            print(vocab)
            q = VocabdictDto.query.filter_by(vocab=vocab).all()
            print(q)
            mylist2=[]
            if not q:
                continue
            for i in q:
                # vocabdict = sel_searching_data(driver, vocab, vocabdict)
                # print(vocabdict)

                 
                mylist2.append(i.meaning)
            vocabdict[vocab]=mylist2
            vocablist.append(vocabdict)
        
        print(vocablist)
        return vocablist

    @classmethod
    def add_vocab(cls, userid, vocabId, newv):
        add_vocab = cls.query.filter(userid == userid, vocabId != vocabId).add(newv)
        return add_vocab

    @staticmethod   
    def bulk():
        service = VocabPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(VocabDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @classmethod
    def delete_vocab(cls, userid, vocabId):
        del_vocab = cls.query.filter(userid == userid, vocabId == vocabId).delete(vocabId)
        return del_vocab
        
    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(VocabDto.vocabId)).one()

parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
parser.add_argument('user_id', type=int, required=True,
                                        help='This field should be a userid')
parser.add_argument('vocabId', type=int, required=True,
                                        help='This field should be a vocabId')
parser.add_argument('vocab', type=str, required=True,
                                        help='This field should be a vocab')
parser.add_argument('correctAvg', type=float, required=True,
                                        help='This field should be a correctAvg')

class Vocab(Resource):
    @staticmethod
    def get(id):
        print(f'Vocab {id} added')
        vocab = VocabDao.find_by_id(id)
        if vocab:
            return vocab, 200
        

class Vocabs(Resource):
    def post(self):
        ud = VocabDao()
        ud.insert_many('vocabs')

    def get(self):
        data = VocabDao.find_all()
        return data, 200

if __name__ == '__main__':
    
    prepro = VocabPro()
    prepro.hook()