**Classify docs in text collections by using multinomial NB:**
1. data source: Text collection(1095 news documents)
2. classify docs into 13 classes (id 1~13)
3. training docs: 13 classes and each class has 15 training documents.
4. The remaining documents are for testing
5. Test the result on kaggles.

**Note:**
For each class, you have to calculate M P(X=t|c) parameters.
M is the size of your vocabulary.
Then, the total number of parameters in your system will be |C|*M  can be a huge number.

We know that many terms in the vocabulary are not indicative.

Employ at least one feature selection method and use only 500 terms in your classification.
Χ2 test.
Likelihood ratio.
Pointwise/expected MI. 
Frequency-based methods.

When classify a testing document, terms not in the selected vocabulary are ignored.
To avoid zero probabilities, calculate P(X=t|c) by using add-one smoothing.

