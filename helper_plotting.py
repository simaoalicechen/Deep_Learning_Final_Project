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
    plt.title(f'Multiple Training Losses at epoch {num_epochs}')
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
def plot_multiple_training_accuracies(real_accuracies, fake_accuracies, num_epochs, save_dir):
    """
    Plot the accuracies of discriminator on real and fake data per batch.

    Parameters:
        real_accuracies (list): List of accuracies for real data per batch.
        fake_accuracies (list): List of accuracies for fake data per batch.
        num_epochs (int): Current epoch number to indicate in the plot title.
        save_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(real_accuracies, label='Real Data Accuracy', color='green', linestyle='-', marker='o')
    plt.plot(fake_accuracies, label='Fake Data Accuracy', color='red', linestyle='-', marker='x')
    plt.title(f'Batch-wise Discriminator Accuracies at Epoch {num_epochs}')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a file
    filename = f'discriminator_accuracies_epoch_{num_epochs}.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename))
    plt.close() 
    print(f"Plot saved as {filename}")