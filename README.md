# Kaggle Project
## Digit Recognizer
Competition link - [Santa Gift Matching(https://www.kaggle.com/c/santa-gift-matching) 
####  Tags: Optimization, Recommendation Systems
### Submitted by Nasrudin Salim
[Kaggle Profile](https://www.kaggle.com/nasdin/) 
#### Score: 50/371
#### Ranking: Top 13%

***The Challenge***

In this competition, you are given a list of 1,000,000 children and their wish lists of 100 gifts. You are also given a list of 1000 gifts, and their list of 1000 good kids that they prefer to give to.

Your goal is to match the list of 1,000,000 children with a gift for each child, and try to make everyone happy. Both the kids and Santa need to be happy - for kids, the higher the gift is on their wish list, the happier, for Santa and his gifts, the higher the child is on the good kids list, the happier Santa is.

A few details to notice:

The first 0.5% (ChildId 0-5000) children are triplets. More particularly, 0, 1, 2 are triplets, 3, 4, 5, are triplets .... 4998, 4999, 5000 are triplets. Triplets need to be given the same gift even though they might have different preferences.
The next 4% (ChildId 5001-45000) children are twins. More particularly, 5001 and 5002 are twins, 5003 and 5004 are twins, .... 44999 and 45000 are twins. Twins need to be given the same gift even though they might have different preferences.
For each GiftId, there are 1000 of them available. There are exactly the same number of gifts available (1000 * 1000 = 1,000,000). You shall not exceed the quantity of 1000 for each GiftId.
You are scored on the Average Normalized Happiness. Please refer to the Evaluation page to see detailed descriptions.