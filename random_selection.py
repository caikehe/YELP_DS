#! /usr/bin/env python

import io, os
import random, json

config = {}
execfile("params.conf", config)

numberOfData = config["totalData"]


inputfile = config["original_dataset"]

outputfile5star = 'data/input/bow/5StarsSamples.json'
outputfile4star = 'data/input/bow/4StarsSamples.json'
outputfile3star = 'data/input/bow/3StarsSamples.json'
outputfile2star = 'data/input/bow/2StarsSamples.json'
outputfile1star = 'data/input/bow/1StarsSamples.json'

#Define how many samples using for every type of data
_bow = config["bow_Samples"]
_data = config["dataset_Samples"]

numberOfSample = _bow+_data

data_Outputfile = 'data/input/' + str(_data) + 'Samples.json'

randomSelectionList = random.sample(xrange(0, numberOfData), numberOfSample)
bow_RandomSelectionList = randomSelectionList[0:_bow]
data_RandomSelectionList = randomSelectionList[_bow:_bow+_data]

for i in xrange(1,6):
    if os.path.isfile('data/input/bow/' + str(i) + 'StarsSamples.json'):
        os.remove('data/input/bow/' + str(i) + 'StarsSamples.json')

if os.path.isfile(data_Outputfile):
    os.remove(data_Outputfile)

with open(outputfile5star, 'a') as _5starfile, open(outputfile4star, 'a') as _4starfile, open(outputfile3star, 'a') as _3starfile, open(outputfile2star, 'a') as _2starfile, open(outputfile1star, 'a') as _1starfile, open(data_Outputfile, 'a') as data_Outfile, open(inputfile) as inputfileobject:
    for i, line in enumerate(inputfileobject):
        #print(i)
        if i in data_RandomSelectionList:
               data_Outfile.write(unicode(line))        
        elif i in bow_RandomSelectionList:
            if line == '\n':
                break
            data = json.loads(line)
            if data["text"] =="":
                break
            if (data["stars"] == 5):
                _5starfile.write((data["text"].encode('utf-8')))
            elif (data["stars"] == 4):
                _4starfile.write((data["text"].encode('utf-8')))
            elif (data["stars"] == 3):
                _3starfile.write((data["text"].encode('utf-8')))
            elif (data["stars"] == 2):
                _2starfile.write((data["text"].encode('utf-8')))
            elif (data["stars"] == 1):
                _1starfile.write((data["text"].encode('utf-8')))           

print("Finish selecting data. Now you can move to the feature extraction part")
