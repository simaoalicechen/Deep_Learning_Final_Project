import os
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import torch
import torch.nn.functional as F
import torchvision
import torch.autograd
from helper_plotting import plot_multiple_training_losses
# from helper_evaluate import compute_accuracy
# from helper_evaluate import compute_epoch_loss_classifier
# from helper_evaluate import compute_epoch_loss_autoencoder

def train_gan_v1(num_epochs, model, optimizer_gen, optimizer_discr, 
                 latent_dim, device, train_loader, loss_fn=None,
                 logging_interval=200, 
                 save_model=None, save_images_dir = None, lr_scheduler=None):
    num_epochs += 1
    log_dict = {'train_generator_loss_per_batch': [],
                'train_discriminator_loss_per_batch': [],
                'train_discriminator_real_acc_per_batch': [],
                'train_discriminator_fake_acc_per_batch': [],
                'images_from_noise_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    start_time = time.time()
    for epoch in range(1, num_epochs):
        print(epoch)

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            batch_size = features.size(0)

            # real images
            real_images = features.to(device)
            real_labels = torch.ones(batch_size, device=device) # real label = 1

            # generated (fake) images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # format NCHW
            fake_images = model.generator_forward(noise)
            fake_labels = torch.zeros(batch_size, device=device) # fake label = 0
            flipped_fake_labels = real_labels # here, fake label = 1


            # --------------------------
            # Train Discriminator
            # --------------------------

            optimizer_discr.zero_grad()

            # get discriminator loss on real images
            discr_pred_real = model.discriminator_forward(real_images).view(-1) # Nx1 -> N
            real_loss = loss_fn(discr_pred_real, real_labels)
            # real_loss.backward()

            # get discriminator loss on fake images
            discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
            fake_loss = loss_fn(discr_pred_fake, fake_labels)
            # fake_loss.backward()

            # combined loss
            discr_loss = 0.5*(real_loss + fake_loss)
            discr_loss.backward()

            optimizer_discr.step()

            # --------------------------
            # Train Generator
            # --------------------------

            optimizer_gen.zero_grad()

           

            # get discriminator loss on fake images with flipped labels
            discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
            gener_loss = loss_fn(discr_pred_fake, flipped_fake_labels)
            gener_loss.backward()

            optimizer_gen.step()

             # Adjust learning rate if lr_scheduler is provided
            if lr_scheduler is not None:
                lr_scheduler.step()

            # --------------------------
            # Logging
            # --------------------------   
            log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
            log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
            
            predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
            predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
            acc_real = (predicted_labels_real == real_labels).float().mean()*100.
            acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
            log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
            log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item()) 

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                       % (epoch+1, num_epochs, batch_idx, 
                          len(train_loader), gener_loss.item(), discr_loss.item()))

            # Save generated images
            if save_images_dir is not None and not batch_idx % logging_interval:
                with torch.no_grad():
                    fake_images = model.generator_forward(fixed_noise).detach().cpu()
                    torchvision.utils.save_image(fake_images, 
                                                  f'{save_images_dir}/epoch_{epoch}_batch_{batch_idx}.png', 
                                                  nrow=8, normalize=True)

        ### Save images for evaluation
        with torch.no_grad():
            fake_images = model.generator_forward(fixed_noise).detach().cpu()
            log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))


        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

        if save_model is not None:
            torch.save(model.state_dict(), save_model)
            os.makedirs("reports", exist_ok=True)
        if epoch == 1: 
            plot_multiple_training_losses(
                losses_list=(
                    log_dict['train_discriminator_loss_per_batch'],
                    log_dict['train_generator_loss_per_batch']
                ),
                num_epochs= epoch,
                custom_labels_list=(' -- Discriminator', ' -- Generator'),
                save_dir="reports"
            )
        if epoch == 20: 
            plot_multiple_training_losses(
                losses_list=(
                    log_dict['train_discriminator_loss_per_batch'],
                    log_dict['train_generator_loss_per_batch']
                ),
                num_epochs= epoch,
                custom_labels_list=(' -- Discriminator', ' -- Generator'),
                save_dir="reports"
            )
        if epoch == 30: 
            plot_multiple_training_losses(
                losses_list=(
                    log_dict['train_discriminator_loss_per_batch'],
                    log_dict['train_generator_loss_per_batch']
                ),
                num_epochs= epoch,
                custom_labels_list=(' -- Discriminator', ' -- Generator'),
                save_dir="reports"
            )
        if epoch == 40: 
            plot_multiple_training_losses(
                losses_list=(
                    log_dict['train_discriminator_loss_per_batch'],
                    log_dict['train_generator_loss_per_batch']
                ),
                num_epochs= epoch,
                custom_labels_list=(' -- Discriminator', ' -- Generator'),
                save_dir="reports"
            )
        if epoch == 50: 
            plot_multiple_training_losses(
                losses_list=(
                    log_dict['train_discriminator_loss_per_batch'],
                    log_dict['train_generator_loss_per_batch']
                ),
                num_epochs= epoch,
                custom_labels_list=(' -- Discriminator', ' -- Generator'),
                save_dir="reports"
            )
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    return log_dict




