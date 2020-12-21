import pandas as pd
from mangotoeic.ext.db import db ,openSession
from typing import List
from flask_restful import Resource, reqparse
from sqlalchemy import func
import os
import pickle
basedir = os.path.dirname(os.path.abspath(__file__))     
class PredictMFPro:
    def __init__(self):
        ...
    def read_csv(self):
        df=pd.read_csv(os.path.join(basedir, "./data/unstack.csv"))
        with open(os.path.join(basedir,'./data/idx2qId.pickle'),'rb') as f:
            data=pickle.load(f)
        with open(os.path.join(basedir,'./data/idx2userid.pickle'),'rb') as f:
            data2=pickle.load(f)
        return df ,data,data2
    def hook(self):
        df,data,data2=self.read_csv()
        df=self.prepro(df,data,data2)
        return df
    def prepro(self,df,data,data2):
        print(df)
        print(data)
        print(data2)
        df=df.rename(columns={"Unnamed: 0" : "qId", 'Unnamed: 1': 'user_id',"0" :"correctAvg"})
        df.index.names=['id']
        df['qId']=df['qId'].astype(str)
        df['user_id']=df['user_id'].astype(str)
        df=df.replace({"qId":data,"user_id":data2})
        df
        print(df)
        return df

class  PredictMFDto(db.Model):
    __tablename__ ="predictmf"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key =True, index = True)
    qId: int = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    user_id = db.Column(db.Integer,db.ForeignKey('users.user_id'))
    correctAvg = db.Column(db.Float)
    
    def __repr__(self):
        return f'predictmf(id={self.id},qId={self.qId},user_id={self.user_id},correctAvg={self.correctAvg})'

    @property
    def json(self):
        return {
            'qId' : self.qId,
            'user_id' : self.user_id,
            "id":self.id,
            'correctAvg': self.correctAvg
        }

class PredictMFDao(PredictMFDto):
    
    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()
    @staticmethod   
    def bulk():
        service = PredictMFPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(PredictMFDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def save(corpus):
        db.session.add(corpus)
        db.session.commit()
    @classmethod
    def delete(cls,id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()
    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(PredictMFDto.qId)).one()
    @staticmethod
    def pivot_table_build():
        Session =openSession()
        session =Session()
        q=session.query(PredictMFDto)
        df= pd.read_sql(q.statement,q.session.bind)
        df_pivot= df.pivot(index="user_id", columns='qId', values='correctAvg')
        print(df_pivot)
        return df_pivot
class PredictMF(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
        self.dao = PredictMFDao

    def get(self, id):
        item = self.dao.find_by_id(id)
        if item:
            return item.json()
        return {'message': 'Item not found'}, 404

class PredictMF(Resource):
    def get(self):
        ...
if __name__ == '__main__':
    recommend = PredictMFPro()
    recommend.hook()