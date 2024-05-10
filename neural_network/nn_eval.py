import pandas as pd
from models import nn
from sklearn import datasets


if __name__ == "__main__":
    digits = datasets.load_digits(return_X_y=True)
    X, y = digits
    X = X / 16

    print("Evaluating neural network on digits dataset")
    layers = [64, 64, 10]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='multiclass', epochs=128, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [64, 128, 10]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='multiclass', epochs=128, lr=0.5, num_batches=32)
    print(f"(2NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [64, 128, 10]
    decay = 0.01
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='multiclass', epochs=128, lr=0.5, num_batches=32)
    print(f"(3NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [64, 128, 128, 10]
    decay = 0.01
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='multiclass', epochs=128, lr=0.5, num_batches=32)
    print(f"(4NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    nn.make_cost_graph("figures/nn_cost_digits.png", nn.NN(
        [64, 128, 10], decay=0.001), X, y, 0.004, 1, 5)

    df = pd.read_csv('datasets/titanic.csv')
    X, y = df.iloc[:, 1:].drop(columns=["Name"]), df.iloc[:, 0]
    X = nn.one_hot(X, columns=["Pclass", "Sex"])

    print("evaluating neural network on titanic dataset")
    layers = [9, 64, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [9, 64, 2]
    decay = 0.01
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [9, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [9, 128, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    nn.make_cost_graph("figures/nn_cost_titanic.png", nn.NN(
        [9, 64, 2], decay=0.01), X, y, 0.004, 1, 5)

    df = pd.read_csv("datasets/loan.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X = X.drop(columns=["Loan_ID"])
    X = nn.one_hot(X)

    print("evaluating neural network on loan dataset")
    layers = [20, 64, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [20, 64, 2]
    decay = 0.01
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [20, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [20, 128, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    nn.make_cost_graph("figures/nn_cost_loans.png", nn.NN(
        [20, 64, 2], decay=0.01), X, y, 0.004, 5, 5)

    df = pd.read_csv("datasets/parkinsons.csv")
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()

    layers = [22, 64, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [22, 64, 2]
    decay = 0.01
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [22, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    layers = [22, 128, 128, 2]
    decay = 0.001
    res = nn.stratified_k_fold_eval(
        X, y, 10, layers, decay, eval_type='binary', normalize=True, epochs=400, lr=0.5, num_batches=32)
    print(f"(1NN) Acc: {res[1]:.3f}, F1: {res[2]:.3f}")

    nn.make_cost_graph("figures/nn_cost_parkinsons.png", nn.NN(
        [22, 128, 2], decay=0.001), X, y, 0.004, 5, 5)
