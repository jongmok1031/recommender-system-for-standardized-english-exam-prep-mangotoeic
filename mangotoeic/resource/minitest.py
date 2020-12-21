from mangotoeic.ext.db import db,openSession
from mangotoeic.resource.user import UserDto
from flask_restful import Resource, reqparse
import pandas as pd
from flask import request
from mangotoeic.resource.recommendation import RecommendationDao, RecommendationDto
from mangotoeic.resource.predictMF import PredictMFDto 
from mangotoeic.resource.legacy import LegacyDto
from mangotoeic.resource.nextminiset import NextMiniSetDao
import random
class MinitestDto(db.Model):
    __tablename__ = "minitest"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True, index=True)
    qId = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    user_id= db.Column(db.Integer, db.ForeignKey('users.user_id'))
    answer_correctly = db.Column(db.Integer)
    user_avg = db.Column(db.Float)
    user_qid_avg= db.Column(db.Float)
    @property
    def json(self):
        return {
            'user_id' :self.user_id,
            'qId' : self.qId,
            'answer_correctly' :self.answer_correctly,
            "user_avg":self.user_avg
        }
class MinitestDao(MinitestDto):
    
    @staticmethod
    def bulk(body):
        Session = openSession()
        session = Session()
        df=pd.DataFrame.from_dict(body)
        session.bulk_insert_mappings(MinitestDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def get_average():
        Session = openSession()
        session = Session()
        session.execute('update minitest as t inner join (select user_id, avg(answer_correctly) as av from minitest group by user_id ) t1 on t.user_id = t1.user_id set t.user_avg= t1.av;')
        session.commit()
        session.close()
    @staticmethod
    def get_average2():
        Session = openSession()
        session = Session()
        session.execute('update minitest as t inner join (select user_id,qid, avg(answer_correctly) as av from minitest group by user_id, qId ) t1 on t.user_id = t1.user_id and t.qId= t1.qId  set t.user_qid_avg= t1.av;')
        session.commit()
        session.close()
    @staticmethod
    def hook( body):
        users=UserDto.query.all()
        # print(users)
        minvalue= 100
        minuser=None
        for user in users:
            if user.user_id>=17:
                continue
            rcddtos=RecommendationDto.query.filter_by(user_id=user.user_id).all()
            # print(rcddtos)
            mylist=body['qId']
            minis=MinitestDto.query.filter_by(user_id=body['user_id']).all()
            for mini in minis:
                mylist.append(mini.qId)
            
            mylist = set(mylist)
            sum =0
            for rcditm in rcddtos:
                if rcditm.qId in mylist:
                    # print("=="*100,rcditm)
                    minidto =MinitestDto.query.filter_by(user_id=body["user_id"],qId=rcditm.qId).first()
                    # print(minidto)
                    value=abs(rcditm.correctAvg-minidto.user_qid_avg)
                    sum+=value
                elif not rcditm.qId in mylist:
                    sum2=0
                    count=0
                    for rcditm2 in rcddtos:
                        sum2+=rcditm2.correctAvg
                        count+=1
                    avg_of_user_from_rcd= sum2/count
                    minidto =MinitestDto.query.filter_by(user_id=body["user_id"]).first()
                    # print(minidto)
                    value=abs(avg_of_user_from_rcd-minidto.user_avg)
                    sum+=value
            corrent_value=sum
            if corrent_value<minvalue:
                minvalue=corrent_value
                minuser=user.user_id
            
        print(minuser)
        print(minvalue)
        q=PredictMFDto.query.filter_by(user_id=minuser)
        df= pd.read_sql(q.statement,q.session.bind)
        df_sorted_by_values=df.sort_values(by='correctAvg',ascending =False)
        print(df_sorted_by_values)
        p=df_sorted_by_values.iloc[int(len(df)/2)]
        print(p)
        median=p['correctAvg']
        mfdtos=q.all()
        mylist2=[]
        for mfdto in mfdtos:
            # mfdto중 가장 중간 오답률을 찾는다
            difference=median-mfdto.correctAvg
            if abs(difference)>0.5:
                continue
            if difference < 0: #맞출확률이 높다면
                x=random.randint(0,1)
                if x==0: 
                    continue
                if x==1:
                    pass
            mylist2.append(mfdto)
        samplelists= random.sample(mylist2,5)
        mylist3=[]
        print(samplelists)
        for samplelist in samplelists:
            legacydto=LegacyDto.query.filter_by(qId=samplelist.qId).first()
            mylist3.append(legacydto.json)
        print(mylist3)
        NextMiniSetDao.delete(body['user_id'])
        for item in mylist3:
            qId= item['qId']
            
            NextMiniSetDao.add(body['user_id'],qId)
        return mylist3
class Minitest(Resource):
    def post(self):
        pass

class Minitests(Resource):
    @staticmethod
    def post():
        body =request.get_json()
        # body={ "user_id":1 , "qId":[1,2,3,4] ,"answer_correctly":[0,1,1,1] }
        # print(body)
        MinitestDao.bulk(body)
        MinitestDao.get_average()
        MinitestDao.get_average2()
        mylist3=MinitestDao.hook(body)
        # df=RecommendationDao.pivot_table_build()
        return mylist3 , 200


if __name__ == "__main__":
    Minitests.post()

    
        

            
        
               
         



                
                
                    








            

        
        

