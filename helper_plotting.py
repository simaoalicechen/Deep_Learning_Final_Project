import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_multiple_training_losses(losses_list, num_epochs, averaging_iterations=100, custom_labels_list=None, save_dir=None):
    print("num_epochs", num_epochs)
    for i,_ in enumerate(losses_list):
        if not len(losses_list[i]) == len(losses_list[0]):
            raise ValueError('All loss tensors need to have the same number of elements.')
    
    if custom_labels_list is None:
        custom_labels_list = [str(i) for i,_ in enumerate(custom_labels_list)]
    
    iter_per_epoch = len(losses_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    
    for i, minibatch_loss_tensor in enumerate(losses_list):
        ax1.plot(range(len(minibatch_loss_tensor)), minibatch_loss_tensor, label=f'Minibatch Loss{custom_labels_list[i]}')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')

        ax1.plot(np.convolve(minibatch_loss_tensor, np.ones(averaging_iterations,)/averaging_iterations, mode='valid'), color='black')
    
    if len(losses_list[0]) < 1000:
        num_losses = len(losses_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(losses_list[i][num_losses:]) for i,_ in enumerate(losses_list)]
    ax1.set_ylim([0, np.max(maxes)*1.5])
    ax1.legend()

    ###################
    # Set second x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    plt.title(f'Multiple Training Losses in {num_epochs} epochs')
    ###################

    plt.tight_layout()

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # reports_dir = os.path.join(script_dir, "reports") 

    # If save_dir is provided, save the plot to the specified directory
    if save_dir:
        # save_path = os.path.join(reports_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)  # Ensure that the directory exists or create it
        plt.savefig(os.path.join(save_dir, f"training_losses_{num_epochs}.png"))
    else:
        plt.show()

def plot_accuracy_per_epoch(real_acc_per_epoch, fake_acc_per_epoch, num_epochs, save_dir = "reports"):
    plt.figure()
    plt.plot(range(num_epochs), real_acc_per_epoch, label='Real Accuracy')
    plt.plot(range(num_epochs), fake_acc_per_epoch, label='Fake Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Discriminator Accuracy per Epoch in {num_epochs} Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        # save_path = os.path.join(reports_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)  # Ensure that the directory exists or create it
        plt.savefig(os.path.join(save_dir, f"training_acc_{num_epochs}.png"))
    else:
        plt.show()

# May try this later using the same logic with loss plotting
# this keeps track all batches' acc movements
def plot_multiple_training_accuracies(acc_list_real, acc_list_fake, num_epochs, save_dir=None):
    """
    Plot multiple training accuracies (both real and fake) over epochs.

    Args:
    - acc_list_real (list of lists): List containing accuracy values for real samples per epoch.
    - acc_list_fake (list of lists): List containing accuracy values for fake samples per epoch.
    - num_epochs (int): Total number of epochs.
    - save_dir (str): Directory to save the plot. If None, the plot will be displayed.

    Returns:
    None
    """
    # if len(acc_list_real) != len(acc_list_fake):
    #     raise ValueError("The number of lists for real and fake accuracies must be the same.")

    num_lists = len(acc_list_real)

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for i in range(num_lists):
        ax1.plot(range(1, num_epochs + 1), acc_list_real[i], label=f"Real Acc List {i+1}")
        ax1.plot(range(1, num_epochs + 1), acc_list_fake[i], label=f"Fake Acc List {i+1}")

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()

    plt.title(f"Multiple Training Accuracies in {num_epochs} epochs")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"training_accuracies_{num_epochs}.png"))
    else:
        plt.show()