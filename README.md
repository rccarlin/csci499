
- [ ] (5pt) *Report* your results through an .md file in your submission; discuss your implementation choices and document the performance of your model (both training and validation performance) under the conditions you settled on (e.g., what hyperparameters you chose) and discuss why these are a good set.


## Available Bonus Points

You may earn up to 10pt of *bonus points* by implementing the following bells and whistles that explore further 
directions. For these, you will need to compare the performance of the base model against whatever addition you try. 
Add those details to your report. If you implement bonus items, your base code implementing the main assignment must 
remain intact and be runnable still.

- [ ] (*5pt*) Initialize your LSTM embedding layer with [word2vec]
- (https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/), 
- [GLoVE](https://nlp.stanford.edu/projects/glove/), or other pretrained *lexical* embeddings. 
- How does this change affect performance?
- [ ] (*5pt*) The action and the object target may be predicted via independent classification heads. Compare 
- independent classification heads to a setup where the target head also takes in the predictions of the action head, 
- as well as a setup where the action head takes in the predictions of the target head. How do these setups change 
- performance?
- [ ] (*10pt*) Perform a detailed analysis of the data itself or of your model's performance on the data. This bonus is 
- very open ended, and points will be based on the soundness of the implementation as well as the insights gained and 
- written up in the report. For example, you could cluster instructions based on learned embeddings, or even just by 
- looking at token frequencies, and see clusters correspond to action or object targets in any consistent way. You could
- also analyze which training or validation examples your model gets wrong with high confidence (e.g., most probability 
- mass on the wrong choice at prediction time) and see, qualitatively, if you can identify systematic misclassifications 
- of that type and hypothesize about why they happen.
- [ ] (*10pt*) We used a word-level tokenizer. What happens if you try to use a character-level model? Likely the 
- sequences will just be too long! What happens if you swap your LSTM out for a CNN-based model that operates on 
- characters? Note you'll have to write new preprocessing code to try this.


