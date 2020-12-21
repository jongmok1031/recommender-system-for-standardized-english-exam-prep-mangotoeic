from mangotoeic.ext.db import db,openSession
from mangotoeic.resource.user import UserDto
from flask_restful import Resource, reqparse
import pandas as pd
from flask import request
from mangotoeic.resource.recommendation import RecommendationDao, RecommendationDto
from mangotoeic.resource.predictMF import PredictMFDto 
from mangotoeic.resource.legacy import LegacyDto
import random
class NextMiniSetDto(db.Model):
    __tablename__ = "nextminiset"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True, index=True)
    qId = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    user_id= db.Column(db.Integer, db.ForeignKey('users.user_id'))
    @property
    def json(self):
        return {
            'user_id' :self.user_id,
            'qId' : self.qId
        }
class NextMiniSetDao(NextMiniSetDto):
    @staticmethod
    def add( user_id, qId):
        Session = openSession()
        session = Session()
        mini = NextMiniSetDto(qId = qId, user_id=user_id)
        session.add(mini)
        session.commit()
        
    
    @staticmethod
    def delete( user_id):
        # Session = openSession()
        # session = Session()
        minis = NextMiniSetDto.query.filter_by(user_id =user_id).all()
        for mini in minis:
            db.session.delete(mini)
            db.session.commit()

    @staticmethod
    def bulk(body):
        Session = openSession()
        session = Session()
        df=pd.DataFrame.from_dict(body)
        session.bulk_insert_mappings(NextMiniSetDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def find_by_id(id):
        dtos=NextMiniSetDto.query.filter_by(user_id= id).all()   
        mylist=[]
        for dto in dtos:
            legacy=LegacyDto.query.filter_by(qId=dto.qId).first()
            mylist.append(legacy.json)
        return mylist
class NextMiniSet(Resource):
    @staticmethod
    def get(id):
        print(id)
        mylist=NextMiniSetDao.find_by_id(id)
        print(mylist)
        return mylist, 200


        
        