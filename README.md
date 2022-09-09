# Main Project Report

- [ ] (5pt) *Report* your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set.

# Input Analysis for Bonus Points
I want to start by saying thank you for making this a bonus question because I looove doing data analysis. Plus, we
learned in class that a) it's good to see what the computer is given to better understand why it's responding the way
that it is and b) if a human can't find patterns, the computer is unlikely to be able to.

I focused my efforts on 2.5 main questions: 
1. Does the training set have some examples disproportionately represented (i.e., do we have another Count of Monte 
   Cristo on our hands)?
2. Are certain action-target pairs more common than others?
3. For Q1 and Q2, if the answer is yas, do they make sense for the problem at hand or are they challenges for the 
   computer to overcome?


### Question 1
I started by making four command-to-frequency dictionaries using the (for the record, when I say command I mean actions
or targets). One dictionary for train action, one for train target and so on. I feel a little bad for looking at the 
validation set, but ultimately I did it to make sure I wasn't blindly training the model on wildly different tasks 
than I was validating it on. Then I normalized the values of the four dictionaries to make them easier to interpret.

I found that the most common (correct) actions for the training set were GotoLocation (~48%), PickupObject (~22%), and
PutObject (~20%). At first glance, it may be a little worrying to see that only 3/8 of the training actions make up 
roughly 90% of the examples. But then I realized that these are *very* common actions both in the sense that going 
somewhere/ picking something up/ putting something down occur in a ton of tasks/ series of tasks in my daily life *and* 
that these actions are broad enough to be realistically applied to a large number of targets. In contrast, the least
common action in the training set is ToggleObject (<2%), which makes sense because few objects can really be "toggled", 
so toggling should be a less common action.

For the record, the validation set has GotoLocation, PickupObject, and PutObject at practically the same frequency 
(48/22/20), so in this instance the training examples likely prepare the model well for the seen validation. 

The most common targets for the training set were countertop (~9%), diningtable (~6%), sinkbasin (~5%), and fridge 
(~5%). The first thing I noticed was that these are more locations rather than objects. Sure, you could treat the 
fridge as an object and say "unplug the fridge," but I think it's more common to take things in and out of the 
fridge (i.e., it's a location for foodstuffs). I suspect these targets are often paired with GotoLocation (more on 
pairs later). Since GotoLocation is the most common action, it makes sense that targets that best pair with it are
also common. I also want to point out that these are locations that can have a lot of uses; most anything that can
be picked up can be placed on a countertop or diningtable, and most food things can be in the fridge. These targets'
versatility, combined with the fact that they make up only 25% of examples, makes me suspect that there isn't 
a weird disproportionate skew in the training targets.

The most common non-location target in the training set was potato (~3%), soon followed by apple and tomato. Once I saw 
that foodstuffs were the most popular non-location targets, I asked myself, "Is this a cooking robot?" Counters and 
fridges certainly belong in the kitchen, as do some of semi-popular locations like microwave and trashcan. I'm not sure
what to do with this information other than wonder if this means the model will perform better on kitchen tasks?

Oddly enough, some of train's least common targets were glassbottle and pluger, both of which are in my kitchen right 
now lol. But overall, I'm not too concerned with the less common targets because all targets are under 10% anyways...

Out of curiosity, I checked to see if any train frequencies and val frequencies were significantly off. Again, I wasn't
checking to make my model better fit the val set, I just wanted to find a problem that could potentially explain poor
performance. Ignoring targets that were in training and not val, the biggest discrepancies were fridge (off by 1.6%), 
tomato (1.7%), and apple(1.6%). I personally think that's more than close enough, I just think it's interesting that the
areas of greatest difference are the ones in a lot of training examples.



### Question 2





