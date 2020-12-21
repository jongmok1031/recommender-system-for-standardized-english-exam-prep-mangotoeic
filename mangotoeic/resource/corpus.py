from mangotoeic.ext.db import db ,  openSession
from typing import List
from flask_restful import Resource, reqparse
import pandas as pd
class CorpusPro:
    def __init__(self):

        self.fpath ='./data/problemcorpus.csv'
    def hook(self):
        df=self.fileread()
        print(df.head(30))
        return df
    def fileread(self):
        df= pd.read_csv(self.fpath,index_col=False,)
        df= df.set_index(['corId'])
    
        print(df)
        return df
        
class  CorpusDto(db.Model):
    __tablename__ ="corpuses"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    corId= db.Column(db.Integer, primary_key = True, index = True)
    corpus = db.Column(db.VARCHAR(200))
    
       

    def __repr__(self):
        return f'corpuses(corId={self.corId},corpus={self.corpus})'

    @property
    def json(self):
        return {
            'corId' : self.corId,
            'corpus' : self.corpus,
            
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class CorpusDao(CorpusDto):
    
    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()
    @staticmethod   
    def insert_many():
        service = CorpusPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(CorpusDto, df.to_dict(orient="records"))
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

class Corpus(Resource):
    def __init__(self):
        parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
        parser.add_argument('corId', type=float, required=True, help='This field cannot be left blank')
        parser.add_argument('corpus', type=int, required=True, help='Must enter the store id')
        self.dao = CorpusDao

    def get(self, id):
        item = self.dao.find_by_id(id)
        if item:
            return item.json()
        return {'message': 'Item not found'}, 404

class Corpuses(Resource):
    def get(self):
        ...
if __name__ == '__main__':
    pass