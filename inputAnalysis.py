# Okay so I would like to know more about the data that I am feeding into the computer
# Since it was mentioned in class that it is helpful to get a good understanding of twhat the computer is "seeing"

# And as my Stats in R professor once said, "Garbage in, garbage out"

# Things I would like to do:
# study action/ target frequencies, to see if we have another Count of Monte Cristo on our hands
# I would like to do a correlation analysis
# also i would like to see if the percentages are way off for train vs val... not much I can do about it,
# but it could be a source of error


import json
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)


def normalize(dicti, denom):
    newDict = {}
    for key in dicti:
        newDict[key] = round(dicti[key] / denom, 5)
    return newDict


def main():

    file = open('lang_to_sem_data.json')
    langToSem = json.load(file)
    trainUntoken = langToSem["train"]
    valUntoken = langToSem["valid_seen"]

    train2d = list()
    val2d = list()

    # Sees if certain actions and targets are more common than others
    trainActDict = {}
    trainTargetDict = {}
    valActDict = {}
    valTargetDict = {}
    numTrain = 0
    numVal = 0


    # Getting the counts of
    for bigSet in trainUntoken:
        for smallSet in bigSet:
            # temp = preprocess_string(smallSet[0])
            # newList = [temp, smallSet[1]]
            # train2d.append(newList)
            if smallSet[1][0] in trainActDict:
                trainActDict[smallSet[1][0]] += 1
            else:
                trainActDict[smallSet[1][0]] = 1

            if smallSet[1][1] in trainTargetDict:
                trainTargetDict[smallSet[1][1]] += 1
            else:
                trainTargetDict[smallSet[1][1]] = 1
            numTrain += 1

    for bigSet in valUntoken:
        for smallSet in bigSet:
            # temp = preprocess_string(smallSet[0])
            # newList = [temp, smallSet[1]]
            # val2d.append(newList)

            if smallSet[1][0] in valActDict:
                valActDict[smallSet[1][0]] += 1
            else:
                valActDict[smallSet[1][0]] = 1

            if smallSet[1][1] in valTargetDict:
                valTargetDict[smallSet[1][1]] += 1
            else:
                valTargetDict[smallSet[1][1]] = 1

            numVal += 1

    # how many different targets? how many different actions?
    print(len(valTargetDict))  # 76 targets
    print(len(valActDict))  # 8 actions
    print(len(trainTargetDict))  # 80 targets
    print(len(trainActDict))  # 8 actions

    # now I want to convert to %, bc clearly the training is going to have more instances than the validation...
    # In case I want the raw numbers later, I will make new Dictionaries

    pertrainActDict = normalize(trainActDict, numTrain)
    pertrainTargetDict = normalize(trainTargetDict, numTrain)
    pervalActDict = normalize(valActDict, numVal)
    pervalTargetDict = normalize(valTargetDict, numVal)

    print(valActDict)
    print("\n")
    print(pervalActDict)




    # make a heat map with size numactions * numtargets


main()