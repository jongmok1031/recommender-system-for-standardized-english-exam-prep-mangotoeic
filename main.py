from flask import Flask, request
from flask_restful import Api
from mangotoeic.ext.db import url, db
from mangotoeic.ext.routes import initialize_routes
from mangotoeic.resource.legacy import LegacyDao 
from flask_cors import CORS
from mangotoeic.resource.review import ReviewDao
from mangotoeic.resource.user import User, Users, UserDto, UserDao
from mangotoeic.resource.testresult import TestResultDao, TestResultDto, TestResult, TestResults, Lgbm
from mangotoeic.resource.vocablist import VocablistDto, VocablistDao
from mangotoeic.resource.vocab import VocabDao 
from mangotoeic.resource.vocabdict import VocabdictDao
from mangotoeic.resource.recommendation import RecommendationDao
from mangotoeic.resource.selectedq import SelectedQDao
from mangotoeic.resource.predictMF import PredictMFDao
from mangotoeic.resource.predictvocab import PredictVocabDao
from mangotoeic.resource.minitest import Minitests
from mangotoeic.resource.vocabrcm import VocabRcdDao
import json

app = Flask(__name__)
CORS(app, resources={r'/api/*': {"origins": "*"}})
print(url)

app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

db.init_app(app)
api = Api(app)


@app.route('/api/minitest')
def minitest():
    with open('./mangotoeic/legacy/data/toeic_test.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        json_file.close()
    return json_data

@app.route('/api/diagnosis')
def diagnosis():
    with open('./mangotoeic/resource/data/diagnosis.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        json_file.close()
    return json_data

with app.app_context():
    db.create_all()
    legacy_count = LegacyDao.count()
    print(f'***** Legacies Total Count is {legacy_count} *****')
    if legacy_count[0] == 0:
        LegacyDao.bulk()
    selectedq_count = SelectedQDao.count()
    print(f'***** SelectedQ Total Count is {selectedq_count} *****')
    if selectedq_count[0] == 0:
        SelectedQDao.bulk()

    vocablist_count = VocablistDao.count()
    print(f'***** VocabList Total Count is {vocablist_count} *****')
    if vocablist_count[0] == 0:
        VocablistDao.bulk()
    
    vocabdict_count = VocabdictDao.count()
    print(f'***** VocabDict Total Count is {vocabdict_count} *****')
    if vocabdict_count[0] == 0:
        VocabdictDao.bulk()
    
    vocab_count = VocabDao.count()
    print(f'***** Vocab Total Count is {vocab_count} *****')
    if vocab_count[0] == 0:
        VocabDao.bulk()
    
    vocabpredict_count = PredictVocabDao.count()
    print(f'***** Vocab Predict Total Count is {vocabpredict_count} *****')
    if vocabpredict_count[0] == 0:
        PredictVocabDao.bulk()

    user_count = UserDao.count()
    print(f'***** Users Total Count is {user_count} *****')
    if user_count[0] == 0:
        UserDao.bulk()

    testresult_count = TestResultDao.count()
    print(f'***** TestResult Total Count is {testresult_count} *****')
    if testresult_count[0] == 0:
        TestResultDao.bulk()
    else: 
        TestResultDao.get_average()
    
    review_count = ReviewDao.count()
    print(f'***** Review Total Count is {review_count} *****')
    if review_count[0] == 0 :
        ReviewDao.insert_many()
    recommendation_count = RecommendationDao.count()
    print(f'*****Recommendation Total Count is {recommendation_count} *****')
    if recommendation_count[0] == 0 :
        RecommendationDao.bulk()
    predict_count = PredictMFDao.count()
    print(f'*****PredictMF Total Count is {predict_count} *****')
    if predict_count[0] == 0 :
        PredictMFDao.bulk()
initialize_routes(api)
 