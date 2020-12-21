'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
from nltk.corpus import wordnet as wn
import re
import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from mangotoeic.txtgenerator.GPT2.model import GPT2LMHeadModel
from mangotoeic.txtgenerator.GPT2.utils import load_weight
from mangotoeic.txtgenerator.GPT2.config import GPT2Config
from mangotoeic.txtgenerator.GPT2.sample import sample_sequence
from mangotoeic.txtgenerator.GPT2.encoder import get_encoder
import nltk
from nltk.tokenize import word_tokenize
def text_generator(state_dict,text1):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(text1)
    context_tokens = enc.encode(text1)
    mytext=[]
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            mytext.append(text)
            # print(text)
    return set(mytext)
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
            
            samplelist=random.sample(set(i3),3)
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
            
            
if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text=text_generator(state_dict,"Hi my name is Yeo Won.")
        prelist=seperate_txt(text)  
        text_to_newq(prelist)
        
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
