from mangotoeic.ext.db import db ,openSession
from typing import List
from flask_restful import Resource 
import os
import sys
import torch
from flask import request
import random
import argparse
import numpy as np
from mangotoeic.txtgenerator.main import text_generator
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import re
parentdir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class NewQPro:
    def __init__(self):
        ...

class  NewQDto(db.Model):
    __tablename__ ="newQs"
    __table_args__={'mysql_collate':'utf8_general_ci'}
    qId = db.Column(db.Integer, primary_key = True, index = True)
    question = db.Column(db.VARCHAR(500))
    ansA = db.Column(db.CHAR(255))
    ansB = db.Column(db.CHAR(255))
    ansC = db.Column(db.CHAR(255))
    ansD = db.Column(db.CHAR(255))
    answer = db.Column(db.CHAR(255))

    
    def __repr__(self):
        return f'newQs(id={self.id},ansA={self.ansA},ansB={self.ansB},ansC={self.ansC},ansD={self.ansD},answer={self.answer},question={self.question})'


        
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

class NewQDao(NewQDto):
    @staticmethod
    def generate_newq(text):
        print(os.path.join(parentdir,'txtgenerator','gpt2-pytorch_model.bin'))
        if os.path.exists(os.path.join(parentdir,'txtgenerator','gpt2-pytorch_model.bin')):
            state_dict = torch.load(os.path.join(parentdir,'txtgenerator','gpt2-pytorch_model.bin'), map_location='cpu' if not torch.cuda.is_available() else None)
            text =text_generator(state_dict,text)
            print( text)
            return text
        else:
            print('Please download gpt2-pytorch_model.bin')
            sys.exit()
    @staticmethod
    def text_to_newq(prelist):
        questions=[]
        ansAs=[]
        ansBs=[]
        ansCs=[]
        ansDs=[]
        answers=[]
        if not prelist:
            return 
        for item in prelist:
            text= word_tokenize(item)
            item_with_pos=nltk.pos_tag(text)
            idxlist=[]
            for idx, item in enumerate(item_with_pos):
                if not (item[1]== "NNP" or item[1]=="NN"):
                    idxlist.append(idx)
            
            len_item=len(idxlist)
            print(len_item)
            randidx=random.randint(0,len_item-1)
            print(randidx)
            answer=item_with_pos[(idxlist[randidx])]
            p=wn.synsets(answer[0])
            i3=[]
            if not p:
                continue
            for i in p:
                i2=i.lemma_names()
                if not i2:
                    continue
                for i in i2:
                    i3.append(i)
            mylist=[]
            mylist.append(answer[0])
            subset=set(i3)-set(mylist)        
            
            
            if len(subset)<3:
                continue
            
            samplelist=random.sample(subset,3)
            
            print(samplelist)
            samplelist.append(answer[0])
            random.shuffle(samplelist)
            
            ansAs.append(samplelist[0])
            ansBs.append(samplelist[1])
            ansCs.append(samplelist[2])
            ansDs.append(samplelist[3])
            
            item_with_pos[(idxlist[randidx])]=("___" ,"UNK")
            print(item_with_pos)
            question= ''
            for item, pos in item_with_pos:
                question+=" "
                question+=item
            questions.append(question)
            answers.append(answer[0])
            print(question)
            print(samplelist[0])
            print(samplelist[1])
            print(samplelist[2])
            print(samplelist[3])
        return questions,ansAs,ansBs,ansCs,ansDs, answers
    @staticmethod
    def hook(text):
        generated_txt_set=NewQDao.generate_newq(text)
        prelist=NewQDao.seperate_txt(generated_txt_set)  
        questions,ansAs,ansBs,ansCs,ansDs, answers =NewQDao.text_to_newq(prelist)
        NewQDao.add(questions, ansAs, ansBs , ansCs , ansDs, answers)
        mylist =[]
        for question in questions:
            newqdto=NewQDto.query.filter_by(question=question).first()
            mylist.append(newqdto.json)
        return mylist
    @staticmethod
    def add(questions, ansAs, ansBs , ansCs , ansDs, answers):
        for question, ansA, ansB, ansC, ansD, answer in zip(questions, ansAs, ansBs , ansCs , ansDs, answers):
            x=NewQDto(question=question,ansA=ansA,ansB=ansB,ansC=ansC,ansD=ansD, answer=answer)
            db.session.add(x)
        db.session.commit()
    @staticmethod
    def seperate_txt(txt_set):
        mylist=list(txt_set)
        prelist=[]
        regex= r'\`\]\['
        for item in mylist:
            
            split_txt_list=item.split('.')
            split_txt_set=set(split_txt_list)
            prelist=[ item.replace('\n',' ') for item in split_txt_set if not len(item)<30]
            prelist=[ re.sub(regex,"",item) for item in prelist]
            
            print(prelist)    
        return prelist
class NewQ(Resource):
    pass
class NewQs(Resource):
    @staticmethod
    def post():
        body=request.get_json()
        print(body)
        text=body['text']
        mylist=NewQDao.hook(text)        
        return mylist , 200
        

if __name__ == '__main__':
    pass