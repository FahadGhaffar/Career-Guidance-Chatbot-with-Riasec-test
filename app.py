from flask import Flask,request,redirect,render_template,url_for,jsonify
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter 
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    intents = json.load(file)

app=Flask(__name__)

data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return random.choice(i['responses'])

            results.pop(0)    

# hi=chat(starts)
@app.route('/',methods=['GET'])
def hello():
    
    return jsonify({"message":hi})

conversations="check"
save=""
lists=[]
report=[]
resultvalue=[0,0,0,0,0,0]
resultlist=[[0,6,14,21,29,31,36],
            [1,10,17,20,25,32,38],
            [2,7,16,22,26,30,40],
            [3,11,12,19,27,33,39],
            [4,9,15,18,28,35,41],
            [5,8,14,23,24,34,37]]
field=["Realistic","Investigative","Artistic","Social","Enterprising","Conventional"]            
@app.route('/',methods=['POST'])
def newhello():
    global save
    global lists
    global resultlist
    hi=request.json['message']
    print(f"hi  {hi}")
    print(f"saved  {save}")
    conversations=hi.lower()
    check=response(hi.lower())
    print(f"check  {check}")
    checks=save
    print(f"checks  {checks}")
    now =response(conversations)
    if hi.lower() == "start":
        lists=[]
        checks=""
        # check=response(hi.lower())
        # lists.append(hi.lower())
        conversations=hi.lower()+" "+checks[14:].lower()
        save=response(conversations)
        now =response(conversations)
    if checks[-9:].lower() == "yes or no":
        print("start")
        lists.append(hi.lower())
        print(lists)
        conversations=hi.lower()+" "+checks[14:].lower()
        save=response(conversations)
        now =response(conversations)
    if hi.lower() == "result":
        now = len(lists)

        label_encoder = LabelEncoder() 
        
        # Encode labels in column 'species'. 
        Y= label_encoder.fit_transform(lists)
        print(Y)
        for i in range(len(resultlist)):
            for j in range(len(resultlist[i])):
                # print(resultlist[i][j])
                resultvalue[i] += Y[resultlist[i][j]]
        print(resultvalue)
        for x in zip(field,resultvalue):
            report.append(x)
        print(report)
        now= str(report) 
        k = Counter(report) 
  
# Finding 3 highest values 
        high = k.most_common(3) 
        print(high)   
        # for test in len(lists):
        #     if "yes" in (lists[0],lists[6],lists[13],lists[21],lists[29],lists[31],lists[36]):
        #         print(f" number ")
        #     else:
        #         pass

    print(f"length {len(lists)}")
    print(f" conversations  {conversations} ")
    return jsonify({"message":now})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)