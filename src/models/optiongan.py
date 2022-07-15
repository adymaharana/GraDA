import os
import torch
import torch.functional as F
from tqdm import tqdm
from transformers import GPT2ForSequenceClassification, get_linear_schedule_with_warmup
from modelling_gpt2 import GPT2ForConditionalGeneration
from utils import get_losses

class OptionGanTrainer():
    def __init__(self, config):

        self.config = config
        self.gen = GPT2ForConditionalGeneration(config.GEN)
        self.disc = GPT2ForSequenceClassification(config.DISC)
        self.init_model()


    def init_model(self):
        if self.config.dis_pretrain:
            print('Load pre-trained discriminator: {}'.format(self.config.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(self.config.pretrained_dis_path, map_location='cuda:{}'.format(self.config.device)))
        if self.config.gen_pretrain:
            print('Load MLE pre-trained generator: {}'.format(self.config.pretrained_gen_path))
            self.gen.load_state_dict(torch.load(self.config.pretrained_gen_path, map_location='cuda:{}'.format(self.config.device)))

        if self.config.CUDA:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def init_training(self):

        self.gen_adv_optimizer = torch.optim.Adam(self.gen.parameters(), lr=self.config.gen_lr)
        self.gen_scheduler = get_linear_schedule_with_warmup(
            self.gen_adv_optimizer, num_warmup_steps=self.config.gen_warmup_steps, num_training_steps=self.config.num_training_steps
        )
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=self.config.disc_lr)
        self.disc_scheduler = get_linear_schedule_with_warmup(
            self.disc_optimizer, num_warmup_steps=self.config.disc_warmup_steps, num_training_steps=self.config.num_training_steps
        )
        self.disc_criterion = torch.nn.CrossEntropyLoss()

    def train(self, data_loader):

        self.init_training()

        # ===ADVERSARIAL TRAINING===
        print('Starting Adversarial Training...')


        for epoch in range(self.config.num_train_epochs):
            for i, batch in tqdm(enumerate(data_loader)):
                print('-----\nEpoch %d\n-----' % epoch)
                self.adv_train_generator(batch)  # Generator
                self.train_discriminator(batch)  # Discriminator

                if i % self.config.save_steps == 0:
                    torch.save(self.gen, os.path.join(self.config.output_dir, 'epoch_%s_%s_textG.pth' % (epoch, i)))
                    torch.save(self.disc, os.path.join(self.config.output_dir, 'epoch_%s_%s_textD.pth' % (epoch, i)))

    def train_discriminator(self, batch):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """

        for step in range(self.config.num_disc_updates):
            # prepare loader for training
            pos_samples = torch.cat((batch['input'], batch['target']), dim=-2)
            neg_samples = self.gen.generate(batch['input'])
            d_loss, train_acc = self.train_dis_epoch(self.disc, (pos_samples, neg_samples))


    def adv_train_generator(self, model, batch):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """

        total_g_loss = 0
        for step in range(self.config.num_gen_updates):

            real_samples = torch.cat((batch['input'], batch['target']), dim=-2)
            gen_samples = self.gen.sample(batch['input'], self.config.batch_size, one_hot=True)
            if self.config.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, self.config.vocab_size).float()

            # ===Train===
            d_out_real = self.disc(real_samples)
            d_out_fake = self.disc(gen_samples)
            gen_loss, _ = get_losses(d_out_real, d_out_fake, self.config.loss_type)
            gen_loss.backward()
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
            self.gen_adv_optimizer.step()

            total_g_loss += gen_loss.item()

        return total_g_loss

    def train_gen_epoch(self, model, data_loader, criterion):
        total_loss = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if self.config.CUDA:
                inp, target = inp.cuda(), target.cuda()

            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.gen_adv_optimizer.zero_grad()
            loss.backward()
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
            self.gen_adv_optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)


    def train_dis_epoch(self, model, data_loader):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if self.config.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = self.disc_criterion(pred, target)
            loss.backward()
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
            self.disc_optimizer.step()

            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc

