# bank-customer-churn-modeling
Given a Bank customer, can we build a classifier which can determine whether they will leave or not?



Analyse the dataset :

First thing to do is looking for relevant features,

Using the code in « analyse,py » we can see the difference in the histograms of some features between people who stayed and those who left :


Age :

![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Age_stayed.png) | ![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Age_left.png)








		stayed								left


Gender : (0 : female, 1 : male)



![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Gender_stayed.png) | ![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Gender_left.png)









		stayed								left







Geography : (0 : France, 1 : Spain, 2 : Germany)

![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Geography_stayed.png) | ![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/Geography_left.png)










		stayed								left


NumOfProducts :



![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/NumOfProducts_stayed.png) | ![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/NumOfProducts_left.png)









		stayed								left

IsActiveMember :




![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/IsActiveMember_stayed.png) | ![alt text](https://github.com/Soufiane-Fartit/bank-customer-churn-modeling/blob/master/analysis/IsActiveMember_left.png)









		stayed								left







Discussion :

it appears that :

-	older people tend to leave more often than young ones
-	females are more likely to leave than males
-	German clients leave more often than French and Spanish ones
-	Clients with only one products are more likely to leave
-	Innactive members have more chances to leave

There’s enough features that provide us with information to predict either a client will leave or not,
after visualising other features they appear to be non- relevant so they wont be included,





Building the classifier and making predictions :

this is a classification problem with less than 100K sample in dataset, the trivial algorithm to chose for this kind of problems is the K-neerest neighbors,

in the file « main,py » the code :
-	load data,
-	drop irrelevant features
-	change text features to numbers,
-	split it to train and test
-	train the K-neighborgs classifer
-	make predictions on the test set

we get 82 % accuracy,



The final answer is YES, we can predict if a costumer is leaving or not,
