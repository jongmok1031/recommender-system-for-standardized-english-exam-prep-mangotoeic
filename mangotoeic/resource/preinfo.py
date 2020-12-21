from flask.globals import session
from flask import request
from flask_restful import Resource
from sqlalchemy import func
from mangotoeic.ext.db import db, openSession
import json
from mangotoeic.resource.user import UserDto

# 토익 시험 몇번 봤는지, 목표점수, 시험날짜, 본인의 영어실력 입력
# 데이터 베이스에 반영할 건지?

class PreInfoDto(db.Model):
    __tablename__ = 'preinfo'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    id = db.Column(db.Integer, primary_key=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey(UserDto.user_id))
    experience = db.Column(db.Integer)
    target_score = db.Column(db.Integer)
    reason = db.Column(db.Integer)
    self_check = db.Column(db.Integer)

    def __repr__(self):
        return f'user_id={self.user_id}, experience={self.experience},\
                target_score={self.target_score}, reason={self.reason},\
                self_check={self.self_check}'


    @property
    def json(self):
        return {
            'id' : self.id,
            'timestamp' : self.timestamp,
            'user_id' : self.user_id,
            'experience' : self.experience,
            'target_score' : self.target_score,
            'reason': self.reason,
            'self_check': self.self_check
        }
    

class PreInfoDao(PreInfoDto):

    def __init__(self):
        pass

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
        return session.query(func.count(PreInfoDto.user_id)).one()
            
    @staticmethod
    def save(preinfo):
        db.session.add(preinfo)
        db.session.commit()

    @staticmethod
    def add_preinfo(data):
        print(data)
        user_id= data['user_id']
        experience = data['diagnosis'][0]
        target_score = data['diagnosis'][1]
        reason = data['diagnosis'][2]
        self_check = data['diagnosis'][3]
        x = PreInfoDto(user_id=user_id, experience=experience, target_score=target_score, reason=reason, self_check=self_check)
        db.session.add(x)
        db.session.commit()

class PreInfo(Resource):

    @staticmethod
    def post():
        body = request.get_json()
        print("+++=++"*30,body)
        PreInfoDao.add_preinfo(body)
class Count(Resource):
    @staticmethod
    def post():
        body = request.get_json()
        userdto=UserDto.query.filter_by(email=body['email']).first()
        user_id= userdto.user_id
        preinfodto=PreInfoDto.query.filter_by(user_id=user_id).first()
        count=0
        if not preinfodto:
            count=0
        if preinfodto:
            count=1
        return count ,200            
# class TestResults(Resource):
#     @staticmethod
#     def post():
#         body = request.get_json()
#         print(body)
#         # df=pd.DataFrame.from_dict(body)
#         TestResultDao.add_testresult(body)

#         return {'id': "good"}, 200


    # def post():
    #     args = parser.parse_args()
    #     print(f'User {args["id"]} added ')
    #     params = json.loads(request.get_data(), encoding='utf-8')
    #     if len(params) == 0:

    #         return 'No parameter'

    #     params_str = ''
    #     for key in params.keys():
    #         params_str += 'key: {}, value: {}<br>'.format(key, params[key])
    #     return {'code':0, 'message': 'SUCCESS'}, 200

    # @staticmethod
    # def get(id: str):
    #     try:
    #         user = TestResultDao.find_by_id(id)
    #         if user:
    #             return user.json()
    #     except Exception as e:
    #         return {'message': 'User not found'}, 404


# if __name__ == "__main__":
#     userdao = UserDao()
#     userdao.userdata_to_sql()
    # userdao.add_user('444', 9834, 3, 1, 39000)
    # userdao.delete_user('115')        
    # userdao.update_user(1, 'user_name', '박지성')
    # userdao.update_user(user_id_loop, names_loop)
    # a = userdao.fetch_user('666')