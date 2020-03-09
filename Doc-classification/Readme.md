**Classify docs in text collections by using multinomial NB:**
1. data source: Text collection(1095 news documents).
2. classify docs into 13 classes (id 1~13).
3. training docs: 13 classes and each class has 15 training documents.
4. The remaining documents are for testing.
5. Test the result on kaggle.

**Note:**
1. For each class, I have to calculate M P(X=t|c) parameters.
2. M is the size of your vocabulary.
3. The total number of parameters in my system will be |C|*M, it can be a huge number.
4. When classify a testing document, terms not in the selected vocabulary are ignored.
5. To avoid zero probabilities, calculate P(X=t|c) by using add-one smoothing.

* Employ at least one feature selection method and use only 500 terms in your classification.
>1. chi-square test.
>2. Likelihood ratio.
>3. Pointwise/expected MI. 
>4. Frequency-based methods.

My method:
1. using Log Liklihood Ratio(LLR)
2. choose the top 38 highest score term from each class(13 times 38 = 494) and thus we can get the feature dictionary.
3. delete the duplicated, remaining 485 unique terms in total.
4. construct the P(t|c) table and do the smmothing at the same time.
5. fit in the multinomila NB classifier

**Results on Kaggle: 98.888%**

