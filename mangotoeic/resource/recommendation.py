import pandas as pd
from mangotoeic.ext.db import db ,openSession
from typing import List
from flask_restful import Resource, reqparse
from sqlalchemy import func
import os
basedir = os.path.dirname(os.path.abspath(__file__))     
class RecommendationPro:
    def __init__(self):
        ...
    def read_csv(self):
        df=pd.read_csv(os.path.join(basedir, "./data/realdata.csv"))
        return df
    def hook(self):
        df=self.read_csv()
        df=self.prepro(df)
        return df
    def prepro(self,df):
        print(df)
        df= df.rename(columns={'answered_correctly': "correctAvg"})
        df=df.drop(['Unnamed: 0'],axis=1)
        df.index.names=['id']
        print(df)
        return df

class  RecommendationDto(db.Model):
    __tablename__ ="recommendation"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key =True, index = True)
    qId: int = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    user_id = db.Column(db.Integer,db.ForeignKey('users.user_id'))
    correctAvg = db.Column(db.Float)
    
    def __repr__(self):
        return f'recommendation(id={self.id},qId={self.qId},user_id={self.user_id},correctAvg={self.correctAvg})'

    @property
    def json(self):
        return {
            'qId' : self.qId,
            'user_id' : self.user_id,
            "id":self.id,
            'correctAvg': self.correctAvg
        }

class RecommendationDao(RecommendationDto):
    
    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()
    @staticmethod   
    def bulk():
        service = RecommendationPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(RecommendationDto, df.to_dict(orient="records"))
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
        return session.query(func.count(RecommendationDto.qId)).one()
    @staticmethod
    def pivot_table_build():
        Session =openSession()
        session =Session()
        q=session.query(RecommendationDto)
        df= pd.read_sql(q.statement,q.session.bind)
        df_pivot= df.pivot(index="user_id", columns='qId', values='correctAvg')
        print(df_pivot)
        return df_pivot
class Recommendation(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
        self.dao = RecommendationDao

    def get(self, id):
        item = self.dao.find_by_id(id)
        if item:
            return item.json()
        return {'message': 'Item not found'}, 404

class Recommendations(Resource):
    def get(self):
        ...
if __name__ == '__main__':
    recommend = RecommendationDao()
    recommend.pivot_table_build()
    