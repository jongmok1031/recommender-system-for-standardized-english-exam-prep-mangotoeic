from mangotoeic.ext.db import db, openSession,engine
from mangotoeic.resource.user import UserDto
from mangotoeic.resource.legacy import LegacyDto
from sqlalchemy.orm import mapper
import pandas as pd
import json
from typing import List
from flask import request, jsonify
from flask_restful import Resource, reqparse
import os
basedir= os.path.dirname(os.path.abspath(__file__))
Session = openSession()
session = Session()

class OdapPro:
    def __init__(self):
        self.fpath =''
    
    def hook(self):
        df=self.fileread()
        print(df.head())
        return df

    def fileread(self):
        df= pd.read_csv(self.fpath,index_col=False,)
        df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(index ={0:'vocabId'})
        
        return df

class OdapDto(db.Model):
    
    __tablename__ = 'odap'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key=True, index=True)
    user_id: int = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    qId: int = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    # legacy_id = db.relationship("LegacyDao", back_populates='odap')  

    
    def __repr__(self):
        return f'user_id={self.user_id}, qId={self.qId}'

    @property
    def json(self):
        return {
            'user_id' : self.user_id,
            'qId' : self.qId
        }

class OdapVo:
    user_id: int = 0
    qId: int = 0

class OdapDao(OdapDto):
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def add_odap(cls, userid, qId, newq):
        add_odap = cls.query.filter(userid == userid, qId != qId).add(newq)
        return add_odap
    
    @classmethod
    def delete_odap(cls, userid, qId):
        del_odap = cls.query.filter(userid == userid, qId == qId).delete(qId)
        return del_odap

    @classmethod
    def add_odap2(cls,data):
        user_id= data['user_id']
        print(user_id)
        for qid in data['qId']:
            dtos= OdapDto.query.filter_by(user_id=user_id).all()
            mylist=[]
            for dto in dtos:
                mylist.append(dto.qId)
            if qid in mylist:
                continue
            some_user=UserDto.query.filter_by(user_id=user_id).first()
            some_question=LegacyDto.query.filter_by(qId=qid).first()
            print(some_question)
            print(some_user)
            
            x=OdapDto(user=some_user, legacy=some_question)
            db.session.add(x)    
        db.session.commit()
    
    @classmethod
    def fetch_all(cls, userid):
        some_user=UserDto.query.filter_by(user_id=userid).first()
        print(some_user)
        return 
    
    @staticmethod   
    def bulk(data):
        
        session.bulk_insert_mappings(OdapDto, data.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @classmethod
    def join(cls):
        for u, a in session.query(LegacyDto,cls).\
                    filter(LegacyDto.qId==cls.qId).\
                    filter(OdapDto.user_id=='16').\
                        all():
            print(u)
            print(a)
        

parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
parser.add_argument('user_id', type=int, required=True,
                                        help='This field should be a userid')
parser.add_argument('qId', type=int, required=True,
                                        help='This field should be a qId')

class Odap(Resource):
    @staticmethod    
    def post():
        args = request.get_json()
        print(args)
        d=UserDto.query.filter_by(user_id=args['user_id']).first()
        print(d.odap)
        blist = []
        for idx, item in enumerate(d.odap):
            p = LegacyDto.query.filter_by(qId=item.qId).first()
            blist.append(p.json)

        return blist , 200
    
    @staticmethod
    def update():
        args = parser.parse_args()
        print(f'Question {args["id"]} updated')
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Question {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class Odaps(Resource):
    def get(self):
        data = OdapDao.fetch_all()
        return data, 200

    def post(self):
        body = request.get_json()

        # print(body)
        # df=pd.DataFrame.from_dict(body)
        OdapDao.add_odap2(body)

        return {'id': "good"}, 200
    
    #{'user_id': None, 'qId': [2, 3, 4]}

        
if __name__ == '__main__':
    # association_table= db.Table('association', db.metadata,db.Column('legacies',db.Integer,db.ForeignKey('legacies.qId')))
    # print(type(association_table))
    # for t in db.metadata.sorted_tables:
    #     print(t.name)
    # for c in association_table.c:
    #     print(c)
    # association_table.create(engine,checkfirst=True)
    # OdapDao.with_parents()
    # # print(OdapDto.legacy)
    dao = OdapDao
    dao.fetch_all(16)
