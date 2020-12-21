from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

db = SQLAlchemy()
Base = declarative_base()

config = {
    # 'user' : 'mangotoeic',
    # 'password' : 'mangotoeic',
    # 'host': 'mangotoeic.cgaqgqvxtixg.ap-northeast-2.rds.amazonaws.com',
    # 'port' : '3306',
    # 'database' : 'mangotoeic'
    'user' : 'root',
    'password' : 'root',
    'host': 'localhost',
    'port' : '3306',
    'database' : 'mariadb'
}
charset = {'utf8':'utf8'}
url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"

engine = create_engine(url)


def openSession():
    return sessionmaker(bind=engine)


