import pandas as pd
from mangotoeic.ext.db import db , openSession
import pandas as pd
from typing import List
from flask import request
from flask_restful import Resource, reqparse
import json
from sqlalchemy import func
import os
basedir= os.path.dirname(os.path.abspath(__file__))
parser = reqparse.RequestParser()  # only allow price changes, no name changes allowed
parser.add_argument('userid', type=str, required=True,
                                        help='This field should be a userid')
parser.add_argument('password', type=str, required=True,
                                        help='This field should be a password')
class LegacyPro:
    def __init__(self):

        self.fpath = os.path.join(basedir,'data/toeic_test.json')
    def hook(self):
        df=self.fileread()
        df=self.filerename(df)
        return df

    def fileread(self):
        df= pd.read_json(self.fpath)
        print(df.transpose())
        # df=df.rename(columns={"Unnamed: 0": "index"})
        # print(df)
        # df=df.set_index(['index'])
        # print(df)
        return df.transpose()
        
    def filerename(self,df):
        df= df.rename(columns={'1':"ansA","2":"ansB","3":"ansC","4":"ansD","question":"question","anwser":"answer"})
        print(df)
        df.index.name= 'qId'
        print(df)
        return df

class  LegacyDto(db.Model):
    __tablename__ ="legacies"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    qId = db.Column(db.Integer, primary_key = True, index = True)
    question = db.Column(db.VARCHAR(300))
    ansA = db.Column(db.CHAR(255))
    ansB = db.Column(db.CHAR(255))
    ansC = db.Column(db.CHAR(255))
    ansD = db.Column(db.CHAR(255))
    answer = db.Column(db.CHAR(255))
    odap = db.relationship("OdapDto", backref='legacy',lazy=True)
    bookmark = db.relationship("BookmarkDto", backref='legacy',lazy=True)
    testresult = db.relationship("TestResultDto", backref='legacy',lazy=True)
    slectedq = db.relationship("SelectedQDto", backref='legacy',lazy=True)    
        
    def __repr__(self):
        return f'legacies(ansA={self.ansA},ansB={self.ansB},ansC={self.ansC},ansD={self.ansD},answer={self.answer},question={self.question},qId ={self.qId})'


        
    @property
    def json(self):
        return {
            'qId' : self.qId,
            'question' : self.question,
            'ansA' : self.ansA,
            'ansB' : self.ansB,
            'ansC' : self.ansC,
            'ansD' : self.ansD,
            'answer' : self.answer
        
        }

class LegacyVo:
    qId:int = 0
    question:str ="" 
    ansA :str = "" 
    ansB :str = ""
    ansC :str = ""
    ansD :str = ""
    answer:str  = ""        

class LegacyDao(LegacyDto):
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls, id):
        return cls.query.filter_by(id == id).first()
    @staticmethod   
    def bulk():
        service = LegacyPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(LegacyDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def save(legacy):
        db.session.add(legacy)
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
        return session.query(func.count(LegacyDto.qId)).one()

class Legacy(Resource):
    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'User {args["id"]} added ')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len(params) == 0:

            return 'No parameter'

        params_str = ''
        for key in params.keys():
            params_str += 'key: {}, value: {}<br>'.format(key, params[key])
        return {'code':0, 'message': 'SUCCESS'}, 200
    @staticmethod
    def get(id):
        print(f'User {id} added ')
        try:
            user = LegacyDao.find_by_id(id)
            if user:
                return user.json()
        except Exception as e:
            return {'message': 'User not found'}, 404

    @staticmethod
    def update():
        args = parser.parse_args()
        print(f'User {args["id"]} updated ')
        return {'code':0, 'message': 'SUCCESS'}, 200

    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'User {args["id"]} deleted')
        return {'code' : 0, 'message' : 'SUCCESS'}, 200

class Legacies(Resource):
    def post(self):
        ud = LegacyDao()
        ud.insert_many('users')

    def get(self):
        print('========== 10 ==========')
        data = LegacyDao.find_all()
        return data, 200


if __name__ == '__main__':
    # prepro = LegacyPro()
    # prepro.hook()
    input_table = LegacyDao
    input_table.with_parents()
    