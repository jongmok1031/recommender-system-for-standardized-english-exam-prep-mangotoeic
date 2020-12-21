import gc
import lightgbm
from flask.globals import session
import pandas as pd
from flask import request
from flask_restful import Resource, reqparse
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine, func
from sqlalchemy import engine
from sqlalchemy.orm import sessionmaker
from mangotoeic.ext.db import db, openSession, engine
from mangotoeic.ext.db import Base
from mangotoeic.resource.user import UserDto
from mangotoeic.resource.legacy import LegacyDto
from flask import jsonify
import json
import joblib

# 토익 시험 몇번 봤는지, 목표점수, 시험날짜, 본인의 영어실력 입력
# 데이터 베이스에 반영할 건지?

class TestResultDto(db.Model):
    __tablename__ = 'testresult'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    id = db.Column(db.Integer, primary_key=True, index=True)
    #userid: int = db.Column(db.Integer, db.ForeignKey(UserDto.user_id))
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    timestamp = db.Column(db.Float)
    qId = db.Column(db.Integer, db.ForeignKey('legacies.qId'))
    user_answer = db.Column(db.Integer)
    answered_correctly = db.Column(db.Float)
    prior_question_elapsed_time = db.Column(db.Float)
    prior_question_had_explanation = db.Column(db.Boolean, default=True)
    user_avg = db.Column(db.Float)

    # def __init__(self, id=0, user_id=0, qId=0, user_answer=0, answered_correctly=0.0, prior_question_elapsed_time=0.0, timestamp=0):
    #     self.id = id
    #     self.timestamp = timestamp
    #     self.user_id = user_id
    #     self.qId = qId
    #     self.user_answer = user_answer
    #     self.answered_correctly = answered_correctly
    #     self.prior_question_elapsed_time = prior_question_elapsed_time

    def __repr__(self):
        return f'user_id={self.user_id}, qId={self.qId},\
                user_answer={self.user_answer}, answered_correctly={self.answered_correctly},\
                prior_question_elapsed_time={self.prior_question_elapsed_time}, prior_question_had_explanation={self.prior_question_had_explanation}'


    @property
    def json(self):
        return {
            'id' : self.id,
            'timestamp' : self.timestamp,
            'user_id' : self.user_id,
            'qId' : self.qId,
            'user_answer' : self.user_answer,
            'answered_correctly': self.answered_correctly,
            'prior_question_elapsed_time': self.prior_question_elapsed_time,
            'prior_question_had_explanation':self.prior_question_had_explanation,
            'user_avg' : self.user_avg
        }
        
        
    # def save(self):
    #     Session = openSession()
    #     session = Session()
    #     newUser = UserDto(user_id = user['user_id'], 
    #                             email = user['email'], 
    #                             password = user['password'])
    #     session.add(newUser)
    #     session.commit()

    # @classmethod
    # def delete(cls):
    #     Session = openSession()
    #     session = Session()
    #     data = cls.query.get(user_id)
    #     session.delete(data)
    #     session.commit()
    

class TestResultDao(TestResultDto):

    def __init__(self):
        pass

    @staticmethod
    def bulk():
        Session = openSession()
        session = Session()
        df = pd.read_csv('./mangotoeic/resource/data/user_table_prepro4.csv')
        session.bulk_insert_mappings(TestResultDto, df.to_dict(orient="records"))
        session.commit()
        session.close()
    @staticmethod
    def bulk2(df):
        Session = openSession()
        session = Session()
        session.bulk_insert_mappings(TestResultDto, df.to_dict(orient="records"))
        session.commit()
        session.close()


    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filer_by(name == name).all()

    @classmethod
    def find_by_id(cls, userid):
        return cls.query.filter_by(userid == userid).first()

    @staticmethod
    def count():
        Session = openSession()
        session = Session()
        return session.query(func.count(TestResultDto.user_id)).one()

    @staticmethod
    def get_average():
        Session = openSession()
        session = Session()
        session.execute('update testresult as t inner join (select user_id, avg(answered_correctly) as av from testresult group by user_id ) t1 on t.user_id = t1.user_id set t.user_avg= t1.av;')
        session.commit()
        session.close()
            

    @staticmethod
    def save(testresult):
        db.session.add(testresult)
        db.session.commit()

    def update_user(self, userid, column, value):
        Session = openSession()
        session = Session()
        session.query(UserDto).filter(UserDto.user_id == userid).update({column : value})
        session.commit()

    @staticmethod
    def add_testresult(data):
        df=pd.DataFrame(data)
        # print(df)
        TestResultDao.bulk2(df)
      
class TestResult(Resource):
    @staticmethod
    def get(id):
        # TestResultDao.get_average()
        data=db.session.query(TestResultDto).filter_by(user_id=id).first()
        pred_score = Lgbm.predict(id)
        user_pred_score = pred_score.iloc[0, 0]
        # print(user_pred_score)
        # print(data)
        return [data.user_avg, user_pred_score], 200
    
    
        

class TestResults(Resource):
    @staticmethod
    def post():
        body = request.get_json()
        print(body)
        # df=pd.DataFrame.from_dict(body)
        TestResultDao.add_testresult(body)
        TestResultDao.get_average()

    @staticmethod
    def get():
        Session = openSession()
        session = Session()
        result = session.execute('select avg(user_avg) from testresult;')
        data = result.first()
        result = round(data[0] * 1000)
        return result, 200

class Lgbm():
    features = [
            'mean_user_accuracy', 
            'questions_answered',
            'std_user_accuracy', 
            'median_user_accuracy',
            'skew_user_accuracy',
            'mean_accuracy', 
            'question_asked',
            'std_accuracy', 
            'median_accuracy',
            'prior_question_elapsed_time', 
            'prior_question_had_explanation',
            'skew_accuracy'
        ]
    target = 'answered_correctly'

    @staticmethod
    def data_prepro():
        testresult_df = pd.read_sql_table('testresult', engine.connect())
        testresult_df.rename(columns={'qId':'content_id'}, inplace=True)

        grouped_by_user_df = testresult_df.groupby('user_id')
        user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew']}).copy()
        user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy', 'median_user_accuracy', 'skew_user_accuracy']

        grouped_by_content_df = testresult_df.groupby('content_id')
        content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew'] }).copy()
        content_answers_df.columns = ['mean_accuracy', 'question_asked', 'std_accuracy', 'median_accuracy', 'skew_accuracy']

        del grouped_by_user_df
        del grouped_by_content_df

        gc.collect()


        testresult_df = testresult_df.merge(user_answers_df, how='left', on='user_id')
        testresult_df = testresult_df.merge(content_answers_df, how='left', on='content_id')    


        testresult_df = testresult_df[Lgbm().features + [Lgbm().target]]
        print(testresult_df)
        testresult_df.to_csv('./geunhong4.csv')
        return testresult_df


    @staticmethod
    def fit():
        lgbm = Lgbm()
        testresult_df = lgbm.data_prepro()
        params = {
            'bagging_fraction': 0.5817242323514327,
            'feature_fraction': 0.6884588361650144,
            'learning_rate': 0.42887924851375825, 
            'max_depth': 6,
            'min_child_samples': 946, 
            'min_data_in_leaf': 47, 
            'n_estimators': 169,
            'num_leaves': 29,
            'random_state': 666
        } 
        model = lightgbm.LGBMClassifier(**params)
        load_model = joblib.load('./mangotoeic/resource/data/lgb_test.pkl')
        new_model = model.fit(testresult_df[Lgbm().features], testresult_df[Lgbm().target], init_model=load_model)
        # joblib.dump(new_model, 'lgb_test2.pkl')
        return new_model

    @staticmethod
    def predict(id):
        lgbm = Lgbm()
        new_model = lgbm.fit()
        test_df = pd.read_csv('./mangotoeic/resource/data/example_test6.csv')
        user_answer_df = pd.read_csv('./mangotoeic/resource/data/user_answer_df.csv')
        content_answer_df = pd.read_csv('./mangotoeic/resource/data/content_answer_df.csv')
        test_df = test_df.merge(user_answer_df, how = 'left', on = 'user_id')
        test_df = test_df.merge(content_answer_df, how = 'left', on = 'content_id')
        test_df['user_id'] = id # 리액트에서 받아오는 user_id로 변경해야함
        # print(test_df)
        test_df['answered_correctly'] = new_model.predict_proba(test_df[Lgbm().features])[:,1]
        test_df = test_df[['row_id', 'user_id', 'content_id', 'answered_correctly']]
        result_group_by = test_df.groupby('user_id')
        result_groupby_user_answer = result_group_by.agg({'answered_correctly': ['mean']}).copy()
        result_groupby_user_answer.columns = ['mean_user_accuracy']
        # print(result_groupby_user_answer)
        return result_groupby_user_answer

    @staticmethod
    def get(id: str):
        try:
            user = TestResultDao.find_by_id(id)
            if user:
                return user.json()
        except Exception as e:
            return {'message': 'User not found'}, 404