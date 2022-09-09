This readme is so long because I am very passionate about the analysis question and I wanted to make up for the 
fact that I have very little faith in my model's ability to compile.

# Main Project Report
Unfortunately my model does not work so I can't do too deep of an analysis on its performance (I eventually decided to 
cut my losses and do analysis on things that didn't require my model to work. Even though it doesn't run, I still 
made some decisions:

- Set learning rate to .15 because that's where my AI fromCSCI 360 performed the best. Obviously other params make the 
  lr preform better or worse, but I figured this was a good starting point. 


# Input Analysis for Bonus Points
I want to start by saying thank you for making this a bonus question because I looove doing data analysis. Plus, we
learned in class that a) it's good to see what the computer is given to better understand why it's responding the way
that it is and b) if a human can't find patterns, the computer is unlikely to be able to.

I focused my efforts on 2.5 main questions: 
1. Does the training set have some examples disproportionately represented (i.e., do we have another Count of Monte 
   Cristo on our hands)?
2. Are certain action-target pairs more common than others?
3. For Q1 and Q2, if the answer is yes, do they make sense for the problem at hand or are they challenges for the 
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
At first, I wanted to make a heat map and then I realized I only knew how to do that in R and I didn't think all that 
setup would be super worth my time... so I made something uglier that will serve the same purpose. I counted each 
instance of each action-target pair and stored these counts in a 2D array. The columns corresponded to actions
while the rows corresponded to targets (this orientation was easier to print out). Then I normalized the values so
each index held the % of occurrences for easier interpretation.

Unfortunately, the arrays are quite large and ugly so I will not paste them in this readme, but the code to make them is
at the bottom of inputAnalysis.py. 

#### A few findings from the Training "Heat Map"
1. Obvious observation, but these distributions are NOT uniform; there are plenty of pairs that never show up (and some 
   pairs show up way more than others). One could argue that this is merely a product of not enough data. These gaps 
   make sense, however, when we consider the context of the problem. Some possible action-target pairs just make no 
   sense or are even impossible. Maybe I don't have enough faith in the model, but I would hate for it to learn 
   something weird because of some improbable examples in the training set.
2. Column 7 corresponds to ToggleObject and is unsurprisingly fairly empty (see above for why). All of col7's weight 
   is in [floorlamp][ToggleObject] (.68%) and [desklamp][ToggleObject] (.88%). This suggests that ToggleObject 
   might imply the target is a lamp (I would do probability analysis if this readme wasn't already too long). But is the
   reverse true? If I am working with a lamp, am I more likely to toggle it than other actions? Logically the answer is
   yes: I am more likely to turn off a lamp than I am to heat up a lamp. But does the data reflect this? Again, yes: 
   for both types of lamps, the only other weight is under the GoToLocation column (actually roughly 50% GoTo 50%
   Toggle). In this instance, knowledge of the target can inform knowledge of the action and vice versa.
3. On the opposite end of the spectrum, there are many positive values in the GotoLocation column. Since there are so 
   many targets with this action, knowing the action isn't too helpful when trying to figure out the target (unlike the
   last example). The reverse isn't very helpful either; there are very few targets that are only GotoLocation. In fact,
   I could only fine one instance of this (handtowelholder)... but I think this *is* a case of not enough data since I
   can imagine other actions with this object. A similar phenomenon happens in the PickupObject and PutObject columns,
   just to a less severe degree. I could have predicited this given what I found from part 1, but it's nice to know 
   they are popular because they are applied to a wide variety of objects as opposed to just a few popular objects
   (which admittedly could make knowing these actions helpful for detrmining the target).
4. Overall, I believe there are enough clusters* to justify having the model learn action-target pairs as opposed to 
   learning the commands separately. *enough actions with few possible targets and enough targets with few possible 
   actions
   
         








