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


# assuming train is first
def printOutliers(dict1, dict2):
    for k1 in dict1:
        if k1 in dict2:
            if abs(dict1[k1] - dict2[k1]) > .01:
                print(k1, "train:", dict1[k1], "val:", dict2[k1])
        #else:
            # print(k1, "is not in val")


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


    # Getting the counts of individual commands (actions or tokens) from both sets
    # Training
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

    # Validation
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

    # print(valActDict)
    # print("\n")
    # print(pervalActDict)

    # Most popular validation actions: GotoLocation, PickupObject, PutObject
    # I am not surprised by those at all, those are pretty basic commands-- you can do them to many types objects
    # (potentially more so than the other actions like heat or toggle) and these actions are common in strings of
    # more complicated series of commands (if you are doing something with an object, you probably have to pick it up).
    # All this to say I think these make sense for the most common --> worthwhile to practice more on more common things

    # Least popular val actions: Toggle, clean, heat --> again, makes sense there's fewer of those because I can think
    # of a lot of stuff that can't be toggled / shouldn't be heated up

    # Most popular validation objects: countertop, diningtable, fridge, sinkbasin: Most common objects are places...
    # guess that makes sense because actions often have specific locations you do them in? Also it was less obvious
    # when I was just looking at the list of objects, but is this system mostly being tested on kitchen things? Is
    # this a cooking robot? If the system performs better on kitchen related activites than not, I wouldn't be horribly
    # surprised. That being said, these 4 objects are super low % so I don't think it's *that* skewed

    # As expected, least common val object are non kitchen things like statue and watercan

    # Most popular train actions: goto, pickup, put. Same as before. Logically makese sense as stated earlier. Glad
    # it's lined up with val so far.

    # Least popular train actions: toggle, clean, cool. Almost the same as val... not too worried because their
    # percentages are almost identical.

    # Most popular train locations: countertop, diningtable, sink, fridge. Out of order but still at the top--
    # nothing to be worried about yet... No one thing is dominating the % which is good

    # FIXME... maybe should have been looking also to see if skewed

    # least popular train objects: glassbottle, plunger, winebottle... little concerning that some of the least common
    # examples *are* kitchen stuff... I mean I know we shouldn't peak at the validation, but like if the intention is
    # to use this system in a kitchen, shouldn't that inform what examples to give it?

    # this is how I'm quickly filtering to find things to talk about
    # for k in trainTargetDict:
    #     if trainTargetDict[k] <700:
    #         print(k, trainTargetDict[k], pertrainTargetDict[k])


    # now to see if the % are way off
    print("Actions:")
    printOutliers(pertrainActDict, pervalActDict)
    print("\n\nTargets")
    printOutliers(pertrainTargetDict, pervalTargetDict)

    # nothing was more than 10% off, except for things that weren't tested in validation... like the salt and pepper
    # shakers *thinking emoji*

    # Fridge, tomato, sidetable, apple, and bread are all off by >1% but <5%. I may be wrong, but that seems close
    # enough for me

    # and the action distributions are always close

    # make a heat map with size numactions * numtargets


main()