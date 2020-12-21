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

class BookmarkPro:
    def __init__(self):
        self.fpath =''

class BookmarkDto(db.Model):
    
    __tablename__ = 'bookmark'
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id: int = db.Column(db.Integer, primary_key=True, index=True)
    user_id: int = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    qId: int = db.Column(db.Integer, db.ForeignKey('legacies.qId')) 

    def __repr__(self):
        return f'user_id={self.user_id}, qId={self.qId}'

    @property
    def json(self):
        return {
            'user_id' : self.user_id,
            'qId' : self.qId
        }

class BookmarkVo:
    user_id: int = 0
    qId: int = 0

class BookmarkDao(BookmarkDto):
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))
    
    @classmethod
    def delete_bookmark(cls, user_id, qId):
        x=BookmarkDto.query.filter_by(user_id = user_id, qId = qId).one()
        db.session.delete(x)    
        db.session.commit()
    @classmethod
    def add_bookmark(cls,user_id,qId):
        x=BookmarkDto(user_id=user_id, qId=qId)
        db.session.add(x)    
        db.session.commit()
    
    @classmethod
    def fetch_all(cls, userid):
        bookmarkdtos=BookmarkDto.query.filter_by(user_id=userid).all()
        mylist =[]
        for item in bookmarkdtos:
            legacydto=LegacyDto.query.filter_by(qId=item.qId).first()
            mylist.append(legacydto.json)
        return mylist
    @classmethod
    def fetch_all_to_odap(cls, userid):
        bookmarkdtos=BookmarkDto.query.filter_by(user_id=userid).all()
        mylist =[]
        for item in bookmarkdtos:
            mylist.append(item.qId)
        return mylist
    @staticmethod   
    def bulk(data):    
        session.bulk_insert_mappings(BookmarkDto, data.to_dict(orient="records"))
        session.commit()
        session.close()
    
    @classmethod
    def join(cls):
        for u, a in session.query(LegacyDto,cls).\
                    filter(LegacyDto.qId==cls.qId).\
                    filter(BookmarkDto.user_id=='16').\
                        all():
            print(u)
            print(a)

parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
parser.add_argument('user_id', type=int, required=True,
                                        help='This field should be a userid')
parser.add_argument('qId', type=int, required=True,
                                        help='This field should be a qId')

class Bookmark(Resource):
    @staticmethod    
    def post():
        args = request.get_json()
        print(args)
        d=BookmarkDto.query.filter_by(user_id=args['user_id'],qId=args['qId']).first()
        if not d:
            print("None",d)
            BookmarkDao.add_bookmark(args['user_id'],args['qId'])
        if d:
            print("True",d)
            BookmarkDao.delete_bookmark(args['user_id'],args['qId'])
        # bookmarkdtos=BookmarkDto.query.all()
        # mylist =[]
        # for dto in bookmarkdtos:
        #     mylist.append(dto.json)
        data= BookmarkDao.fetch_all_to_odap(args['user_id'])
        print(data)
        return data , 200
    
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

class Bookmarks(Resource):
    @staticmethod
    def get(id):
        data = BookmarkDao.fetch_all(id)
        return data, 200

    def post(self):
        body = request.get_json()
        BookmarkDao.add_odap2(body)
        
        return {'id': "good"}, 200
class BookmarksToOdap(Resource):
    @staticmethod
    def get(id):
     
        data= BookmarkDao.fetch_all_to_odap(id)
        return data, 200    
    #{'user_id': None, 'qId': [2, 3, 4]}        
if __name__ == '__main__':
    dao = BookmarkDao
    dao.fetch_all(16)