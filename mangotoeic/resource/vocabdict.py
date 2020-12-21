import pandas as pd
import json
from mangotoeic.ext.db import db, openSession
from flask_restful import Resource, reqparse
import pickle
import os
import numpy as np
from sqlalchemy import func
from mangotoeic.resource.vocablist import VocablistDto
basedir= os.path.dirname(os.path.abspath(__file__))

class VocabdictPro:
    def __init__(self):
        self.fpath = os.path.join(basedir, './data/vocabdict.pickle')
        self.fpath2 = os.path.join(basedir, './data/vocabdict2.pickle')
        self.fpath3 = os.path.join(basedir, './data/vocabdict3.pickle')
        self.fpath4 = os.path.join(basedir, './data/vocabdict4.pickle')
        self.fpath5 = os.path.join(basedir, './data/vocabdict5.pickle')
        self.fpath6 = os.path.join(basedir, './data/vocabdict6.pickle')
    
    def hook(self):
        mylist=self.fileread()
        return mylist
    
    def fileread(self):
        with open(self.fpath, 'rb') as f:
            data = pickle.load(f)
        with open(self.fpath2, 'rb') as f:
            data2 = pickle.load(f)
        with open(self.fpath3, 'rb') as f:
            data3 = pickle.load(f)
        with open(self.fpath4, 'rb') as f:
            data4 = pickle.load(f)
        with open(self.fpath5, 'rb') as f:
            data5 = pickle.load(f)
        with open(self.fpath6, 'rb') as f:
            data6 = pickle.load(f)
        # vlist = ([str(elem) for elem in data.values()])
        # vlist2 = ([str(elem) for elem in data2.values()])
        # vlist3 = ([str(elem) for elem in data3.values()])
        df = pd.DataFrame.from_dict(data, orient='index')
        df2 = pd.DataFrame.from_dict(data2, orient='index')
        df3 = pd.DataFrame.from_dict(data3, orient='index')
        df4 = pd.DataFrame.from_dict(data4, orient='index')
        df5 = pd.DataFrame.from_dict(data5, orient='index')
        df6 = pd.DataFrame.from_dict(data6, orient='index')
        df = df.append(df2)
        df = df.append(df3)
        df = df.append(df4)
        df = df.append(df5)
        df = df.append(df6)
        # print(df)
        mylist=[]
        df.apply(lambda x: mylist.append(x))
        
        return mylist

class VocabdictDto(db.Model):
    
    __tablename__ = 'vocabdict'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True, index=True)
    vocab = db.Column(db.String(50), db.ForeignKey('vocablist.vocab'))
    meaning = db.Column(db.String(300))


    def __repr__(self):
        return f' vocab={self.vocab}, meaning={self.meaning}'
    
    @property
    def json(self):
        return {            
            'vocab': self.vocab,
            'meaning': self.meaning
        }

class VocabdictVo:
    vocab: str = ''
    meaning: str = ''

class VocabdictDao(VocabdictDto):
    @staticmethod   
    def bulk():
        service = VocabdictPro()
        Session = openSession()
        session = Session()
        mylist = service.hook()
        for i,item in enumerate(mylist):
            df2 = item.to_frame()
            df2=df2.rename( columns={i: "meaning"})
            # print(dir(df2.index.names))
            # print(help(df2.index.names))
            df2.index.names=['vocab']
            df2=df2.reset_index()
            df2.index.names=['id']
            df2.replace({None: np.nan }, inplace=True)
            df2=df2.dropna()
            print(df2)
            session.bulk_insert_mappings(VocabdictDto, df2.to_dict(orient="records"))
        session.commit()
        session.close()
   
    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(VocabdictDto.vocab)).one()


parser = reqparse.RequestParser()
parser.add_argument('vocab', type=str, required=True,
                                        help='This field should be a userid')
parser.add_argument('meaning', type=str, required=True,
                                        help='This field should be a vocabId')

class Vocabdict(Resource):
    def get(id):
        pass

if __name__ == "__main__":
    prepro = VocabdictDao
    prepro.bulk()