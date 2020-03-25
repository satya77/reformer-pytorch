import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from AGBDataReader import AGBDataReader
from ReformerSiamese import ReformerSiamese

from tqdm import tqdm

from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, PreTrainedTokenizer
from fairseq.optim.adafactor import Adafactor
import os
import json
import logging
from datetime import datetime


#### Just some code to print debug information to stdout

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# read the setting file
with open('setting.json', 'r') as f:
    args = json.load(f)


class ReformerTrainer(object):

    def __init__(self,
                 model,
                 tokenizer,
                 device=None,
                 train_batch_size=8,
                 eval_batch_size=None,
                 tb_writer=True,
                 tb_dir='./tb_logs',
                 log_dir='./logs'):
        """
        Provides an easy to use class for pretraining and evaluating a Reformer Model.

        :param model: (reformer_pytorch.Reformer)
        :param tokenizer: (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
        :param device: provide manual device placement. If None, will default to cuda:0 if available.
        :param tb_writer: (bool) Whether to write to tensorboard or not.
        :param tb_dir: (str) Where to write TB logs to.
        :param log_dir: (str) Where to write generic logs to.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.log_dir = log_dir

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

        if tb_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tb_dir)

        logging.basicConfig(filename=f'{log_dir}/{datetime.now().date()}.log', level=logging.INFO)




    def train(self,
              epochs,
              train_dataloader,
              eval_dataloader,
              log_steps,
              ckpt_steps,
              ckpt_dir=None,
              gradient_accumulation_steps=1):
        """
        Trains the Reformer Model
        :param epochs: The number of times you wish to loop through the dataset.
        :param train_dataloader: (torch.utils.data.DataLoader) The data to train on.
        :param eval_dataloader: (torch.utils.data.DataLoader) The data to evaluate on.
        :param log_steps: The number of steps to iterate before logging.
        :param ckpt_steps: The number of steps to iterate before checkpointing.
        :param ckpt_dir: The directory to save the checkpoints to.
        :param gradient_accumulation_steps: Optional gradient accumulation.
        :return: Total number of steps, total loss, model
        """

        optimizer = Adafactor(self.model.parameters())

        loss_fn = nn.BCELoss()
        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0

        if ckpt_dir is not None:
            assert os.path.isdir(ckpt_dir)
            try:
                logging.info(f'{datetime.now()} | Continuing from checkpoint...')
                self.model.load_state_dict(torch.load(f'{ckpt_dir}/model_state_dict.pt', map_location=self.device))
                optimizer.load_state_dict(torch.load(f'{ckpt_dir}/optimizer_state_dict.pt'))

            except Exception as e:
                logging.info(f'{datetime.now()} | No checkpoint was found | {e}')

        self.model.train()

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
            logging.info(f'{datetime.now()} | Utilizing {self.n_gpu} GPUs')

        self.model.to(self.device)
        logging.info(f'{datetime.now()} | Moved model to: {self.device}')
        logging.info(
            f'{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}')
        logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
        logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

        for epoch in tqdm(range(epochs), desc='Epochs', position=0):
            logging.info(f'{datetime.now()} | Epoch: {epoch}')
            for step, batch in tqdm(enumerate(train_dataloader),
                                    desc='Epoch Iterator',
                                    position=1,
                                    leave=True,
                                    total=len(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                section1, section2, labels = batch
                output = self.model(section1, section2)

                # only calculating loss
                output = output.view(-1)
                labels = labels.view(-1)


                loss = loss_fn(output, labels)

                if gradient_accumulation_steps > 1:
                    loss /= gradient_accumulation_steps

                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                step_loss += loss.item()
                losses[global_steps] = loss.item()
                local_steps += 1
                global_steps += 1

                if global_steps % log_steps == 0:
                    if self.tb_writer:
                        self.writer.add_scalar('Train/Loss', step_loss / local_steps, global_steps)
                        self.writer.close()
                    logging.info(
                        f'''{datetime.now()} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}''')

                    with open(f'{self.log_dir}/train_results.json', 'w') as results_file:
                        json.dump(losses, results_file)
                        results_file.close()
                    step_loss = 0.0
                    local_steps = 0

                if global_steps % ckpt_steps == 0:
                    # evaluating before every checkpoint
                    self.evaluate(eval_dataloader)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

                    logging.info(f'{datetime.now()} | Saved checkpoint to: {ckpt_dir}')

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'{ckpt_dir}/model_state_dict.pt')
        torch.save(optimizer.state_dict(), f'{ckpt_dir}/optimizer_state_dict.pt')

        return self.model
    def evaluate(self, dataloader):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :return: None
        """
        loss_fn = nn.BCELoss()

        if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
            self.model = nn.DataParallel(self.model)

        self.model.eval()
        eval_loss = 0.0
        correct = 0.0
        eval_steps = 0
        total=0.0
        sigmoid= nn.Sigmoid()
        logging.info(f'{datetime.now()} | Evaluating...')
        for step, batch in tqdm(enumerate(dataloader), desc='Evaluating', leave=True, total=len(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            section1, section2, labels = batch


            with torch.no_grad():
                output = self.model(section1, section2)

            output= output.view(-1)
            labels = labels.view(-1)

            tmp_eval_loss = loss_fn(output, labels)

            if self.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            predicted = torch.gt(output, 0.5)
            correct += (predicted == labels).sum().item()
            eval_steps += 1
            total += labels.size(0)

            eval_loss /= eval_steps

            accuracy= 100 * correct /total
            if self.tb_writer:
                self.writer.add_scalar('Eval/Loss', eval_loss, eval_steps)
                self.writer.close()
                self.writer.add_scalar('accuracy',accuracy, eval_steps)
                self.writer.close()
            logging.info(f'{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | accuracy: {accuracy}')

        return None






if __name__ == '__main__':
    # set parameters
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'reformer'
    model_save_name = 'training_agb_' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # Convert the dataset to a DataLoader ready for training
    logging.info("***** Read AGB train dataset ***** ")
    agb_reader = AGBDataReader('data',tokenizer_method="bert")
    train_data = agb_reader.get_examples('train_raw.tsv', max_seq_length=args['max_seq_length'], read_cache=True)
    train_num_labels = agb_reader.get_num_labels()
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args['train_batch_size'])
    logging.info("Done!")

    logging.info("***** Read AGB dev dataset ***** ")
    dev_data = agb_reader.get_examples('dev_raw.tsv', max_seq_length=args['max_seq_length'], read_cache=True)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args['eval_batch_size'])
    logging.info("Done!")

    logging.info(f'''train_dataloader size: {len(train_dataloader.dataset)} | shuffle: {True}
                             eval_dataloader size: {len(dev_dataloader.dataset)} | shuffle: {False}''')

    model = ReformerSiamese(
        num_tokens=agb_reader.tokenizer.vocab_size,
        emb_dim=1024,
        max_seq_len=agb_reader.tokenizer.max_len)
    trainer = ReformerTrainer( model, agb_reader.tokenizer, train_batch_size=args['train_batch_size'], eval_batch_size=args['eval_batch_size'])
    model = trainer.train(epochs=3,
                          train_dataloader=train_dataloader,
                          eval_dataloader=dev_dataloader,
                          log_steps=100,
                          ckpt_steps=500,
                          ckpt_dir=args["save_model_dir"],
                          gradient_accumulation_steps=1)
    torch.save(model, './ckpts/model.bin')
