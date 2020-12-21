import pandas as pd
from mangotoeic.ext.db import db , openSession
import pandas as pd
from typing import List
from flask import request,jsonify
from flask_restful import Resource, reqparse
import json
from sqlalchemy import func 
import os
from mangotoeic.resource.legacy import LegacyDto

basedir = os.path.dirname(os.path.abspath(__file__))
class SelectedQPro:
    def __init__(self):
        self.fpath = os.path.join(basedir,'data/selecteQID.csv') 
    def hook(self):
        df=self.fileread()
        df=self.filerename(df)
        return df

    def fileread(self):
        df= pd.read_csv(self.fpath)
        return df
    def filerename(self,df):
        print(df)
        df=df.set_index(['selected_qid'])
        df["percent_to_selected"]=0
        df=df.drop(columns=["Unnamed: 0"],axis=1)
        df=df.reset_index()
        df.index.names=['id']
        print(df)
        return df
        
class  SelectedQDto(db.Model):
    __tablename__ ="selectedq"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    id =db.Column(db.Integer, primary_key = True, index = True)
    selected_qid = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    percent_to_selected =db.Column(db.Integer)
    @property
    def json(self):
        return {
            'id' : self.id,
            'selected_qid' : self.selected_qid,
            "percent_to_selected":self.percent_to_selected
        }
class SelectedQDao(SelectedQDto):
    @staticmethod
    def bulk():
        service = SelectedQPro()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(SelectedQDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(SelectedQDto.selected_qid)).one()
    @staticmethod
    def give_random_five_problem():
        Session =openSession()
        session =Session()
        f=session.query(SelectedQDto).order_by(func.random()).limit(5)
        # print(f.all())
        mylist =[]
        
        for item in f.all():
            # print(item.json)
            a =LegacyDto.query.filter_by(qId=item.selected_qid).first()
            # print(a.json)

            mylist.append(a.json)
            # print(mylist)
        return mylist

class SelectedQs(Resource):
    @staticmethod
    def get():
        data = SelectedQDao.give_random_five_problem()
        
        # print(data)
        return data, 200
if __name__ == "__main__":
    pro = SelectedQDao()
    pro.give_random_five_problem()