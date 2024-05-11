# 589-final-project

**Authored by:** \
Mihir Thalanki, \
Aaron Tian
## Running the KNN and Random Forest Evaluations
Code present in './src'
See the results on:
```console
./src/knn-rf-digits-script.ipynb
./src/knn-rf-titanic-script.ipynb
./src/knn-rf-loan-script.ipynb
./src/knn-rf-parkinsons-script.ipynb
```

## Running the Neural Network Evaluations
All neural network code is self-contained and can be found in the folder `./neural_network`. 


To run the code, simply run the following **with** `./neural_network` **as the working directory:**

```console
$ python nn_eval.py
```

Model performances are outputted to the terminal and figures are saved to `./neural_network/figures/`.

### Prerequisites:
`Python >= 3.10` is required to run the neural network evaluations. 

Dependencies:
 - scikit-learn
 - pandas
 - numpy
 - matplotlib

