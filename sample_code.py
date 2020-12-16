import matplotlib.pyplot as plt
import numpy as np
from dataloader import load_data, balanced_sampler
from shared import images_to_column_matrix, plot_metrics_std, training_procedure_5, split_into_folds, combine_emotions_in_fold

learning_rates = [0.01, 0.6, 1]
k = 10  # number of folds for CV
p = 6  # number of PCs
epochs = range(50)
show_imgs = True
sanity_check = True
emotions = ['anger', 'happiness']
images_aligned, cnt = load_data()
images_aligned = balanced_sampler(images_aligned, cnt, emotions)
images_aligned_cv = split_into_folds(images_aligned, emotions, k)
img_shape = images_aligned[emotions[0]][0].shape


def run_with_learning_rate(learning_rate):
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    test_accs = []

    for i in range(k):
        val_i = i
        val_fold = images_aligned_cv[val_i]
        xs_val, ys_val = combine_emotions_in_fold(val_fold, emotions)

        test_i = (i + 1) % k
        test_fold = images_aligned_cv[test_i]
        xs_test, ys_test = combine_emotions_in_fold(test_fold, emotions)

        xs_train = []
        ys_train = []
        for j in range(k):
            if j == val_i or j == test_i:
                continue
            j_fold = images_aligned_cv[j]
            if len(xs_train) == 0:
                xs_train = np.concatenate(
                    (j_fold['anger'], j_fold['happiness']))
                ys_train = np.concatenate(
                    ([0] * len(j_fold['anger']), [1] * len(j_fold['happiness'])))
            else:
                xs_train = np.concatenate(
                    (xs_train, j_fold['anger'], j_fold['happiness']))
                ys_train = np.concatenate(
                    (ys_train, [0] * len(j_fold['anger']), [1] * len(j_fold['happiness'])))

        r_perm = np.random.permutation(len(xs_train))
        xs_train = xs_train[r_perm]
        ys_train = ys_train[r_perm]
        xs_train = images_to_column_matrix(xs_train)

        train_accs_k, train_losses_k, val_accs_k, val_losses_k, test_acc_k = \
            training_procedure_5(xs_train, xs_val, xs_test,
                                 ys_train,
                                 ys_val,
                                 ys_test,
                                 learning_rate=learning_rate,
                                 epochs=epochs,
                                 p=p,
                                 img_shape=img_shape,
                                 sanity_check=sanity_check,
                                 show_imgs=show_imgs and i == 0)

        train_accs.append(train_accs_k)
        train_losses.append(train_losses_k)
        val_accs.append(val_accs_k)
        val_losses.append(val_losses_k)
        test_accs.append(test_acc_k)

    # plot validation and train acc/loss
    plot_metrics_std(epochs, np.array(train_accs), "Accuracy",
                     "accuracy", metric_values_val=np.array(val_accs))
    plot_metrics_std(epochs, np.array(train_losses), "Loss",
                     "loss", metric_values_val=np.array(val_losses))

    # calculate average test accuracy
    avg_test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    print(f"Average test accuracy: {avg_test_acc} ({test_acc_std:.2f})")

    # save training loss for learning rate comparison
    return np.array(train_losses)


fig = plt.figure()
for lr in learning_rates:
    train_losses_lr = run_with_learning_rate(lr)
    plt.errorbar(epochs, train_losses_lr.mean(
        axis=0), yerr=train_losses_lr.std(axis=0))

plt.title("Training losses with various learning rates")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(learning_rates)
fig.show()
