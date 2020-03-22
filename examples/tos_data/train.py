import os
import torch
from AGBDataReader import AGBDataReader
from ReformerSiamese import ReformerSiamese
from datetime import datetime
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import json
from tqdm import tqdm
from tqdm import trange
import numpy as np
import csv

#### Just some code to print debug information to stdout

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# read the setting file
with open('setting.json', 'r') as f:
    args = json.load(f)

#### compute the metrics that we want.
def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    precision, recall, f_score, _ = precision_recall_fscore_support(labels, preds,average='binary')
    return {"precision": precision,
               "recall": recall,
               "f_score": f_score}

#### evaluation
def evaluate(model,eval_dataloader, mode="Dev",steps=0):
    # Eval!
    logger.info("***** Running evaluation {} *****".format(mode))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        section1, section2, labels = batch
        with torch.no_grad():
            y_val = model( section1, section2)

        if preds is None:
            preds = y_val.detach().cpu().numpy()
            out_label_ids = labels.cpu().numpy()
        else:
            preds = np.append(preds, y_val.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    result= compute_metrics(preds, out_label_ids)
    print("Evaluations for "+mode+":")
    print(
        "{0:.3f}\t{1:.3f}\t{2:.3f}".format(
            result["precision"],
            result["recall"],
            result["f_score"],
        )
    )
    #Write them to file so we have them for later
    csv_path = os.path.join(args["output_dir"], "accuracy_evaluation"+mode+"_results.csv")
    csv_headers = [ "steps", "precision","recall","f_score"]
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerow([steps,result["precision"],result["recall"],result["f_score"]])
    else:
        with open(csv_path, mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([steps,result["precision"],result["recall"],result["f_score"]])

#### training
def train(train_dataloader,eval_dataloader, model):

    num_train_optimization_steps = int(
        len(train_dataloader) / args['train_batch_size'] / args['gradient_accumulation_steps']) * args['train_epochs']

    #define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args['train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", num_train_optimization_steps)

    global_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    #epochs
    for _ in trange(int(args['train_epochs']), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        #batchs
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            section1, section2, labels = batch
            y_pred= model(section1, section2)
            loss = criterion(y_pred,labels.view(-1,1))
            #in case we want to do gradient accumulation
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.backward()
            print("\r%f" % loss, end='')

            tr_loss += loss.item()
            nb_tr_examples += section1.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            # Evaluate the model after some steps
            if (step + 1) % args['logging_steps'] == 0:
                evaluate(model,eval_dataloader,steps=global_step)

    return global_step, tr_loss / global_step



# set parameters
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'reformer'
model_save_name = 'training_agb_'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")




# Convert the dataset to a DataLoader ready for training
logging.info("***** Read AGB train dataset ***** ")
agb_reader = AGBDataReader('data')
train_data=agb_reader.get_examples('train_raw.tsv',max_seq_length=args['emb_dim'],read_cache=True)
train_num_labels = agb_reader.get_num_labels()
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args['train_batch_size'])
logging.info("Done!")

logging.info("***** Read AGB dev dataset ***** ")
dev_data=agb_reader.get_examples('dev_raw.tsv',max_seq_length=args['emb_dim'],read_cache=True)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args['eval_batch_size'])
logging.info("Done!")



# Define the model and start training
model=ReformerSiamese(device=device,emb_dim=args['emb_dim'])
global_step, tr_loss = train(train_dataloader,dev_dataloader, model)

#save model after training
output_model_file = os.path.join(args['output_dir'], model_save_name)
logger.info("***** global_step = %s, average loss = %s ***** ", global_step, tr_loss)
if not os.path.exists(os.path.join(args['output_dir'])):
    os.makedirs(os.path.join(args['output_dir']))
logger.info("***** Saving model to %s ***** ", args['output_dir']+"/"+model_save_name)
torch.save(model.state_dict(), output_model_file)

# criterion=torch.nn.BCELoss()
# optimizer= torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
# import time
#
# start_time = time.time()
#
# epochs = 1
#
# max_trn_batch = 800
# max_tst_batch = 300
#
# train_losses = []
# test_losses = []
# train_correct = []
# test_correct = []
#
# for i in range(epochs):
#     trn_corr = 0
#     tst_corr = 0
#
#     # Run the training batches
#     for b, (X_train1,X_train2, y_train) in enumerate(train_dataloader):
#         if b == max_trn_batch:
#             break
#         b += 1
#
#         # Apply the model
#         y_pred = model(X_train1,X_train2)
#         loss = criterion(y_pred, y_train)
#
#         # Tally the number of correct predictions
#         predicted = torch.max(y_pred.data, 1)[1]
#         batch_corr = (predicted == y_train).sum()
#         trn_corr += batch_corr
#
#         # Update parameters
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Print interim results
#         if b % 200 == 0:
#             print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/8000]  loss: {loss.item():10.8f}  \
# accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')
#
#         train_losses.append(loss)
#         train_correct.append(trn_corr)
#
#         # Run the testing batches
#         with torch.no_grad():
#             for b, (X_test1,X_test2, y_test) in enumerate(dev_dataloader):
#                 if b == max_tst_batch:
#                     break
#
#                 # Apply the model
#                 y_val = model(X_test1,X_test2)
#
#                 # Tally the number of correct predictions
#                 predicted = torch.max(y_val.data, 1)[1]
#                 tst_corr += (predicted == y_test).sum()
#
#         loss = criterion(y_val, y_test)
#         test_losses.append(loss)
#         test_correct.append(tst_corr)
#
# print(f'\nDuration: {time.time() - start_time:.0f} seconds')  # print the time elapsed