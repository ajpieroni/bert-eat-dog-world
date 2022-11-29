import os
import sys
import json
import torch
import logging
import numpy as np
from torch import nn
from transformers import AutoModelForSequenceClassification
from sklearn.pipeline import Pipeline
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, ProgressBar
from skorch.hf import HuggingfacePretrainedTokenizer
from torch.optim.lr_scheduler import LambdaLR
# from transformers import AutoTokenizer
from skorch.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


# Basic logging configuration
logging.basicConfig(format='%(asctime)s %(message)s')


class BertModule(nn.Module):
    """ BERT model according to Skorch convention """
    def __init__(self, name, num_labels):
        super().__init__()
        self.name = name
        self.num_labels = num_labels
        self.reset_weights()

    def reset_weights(self):
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.name, num_labels=self.num_labels
        )

    def forward(self, **kwargs):
        pred = self.bert(**kwargs)
        return pred.logits


class ActiveBert():
    """ BERT active learner
            METHODS:
                query_loop: a continous loop for performing active learning.
                            Continues until the pool is empty, or until user
                            interrupt.
                lr_schedule: returns the learning rate based on current step
                calc_score: calculates f_1, precision, recall, accuracy and
                            balanced accuracy
                predict: predicts input using model
            VARIABLES:
                dl: dataloader containing all datasets
                BATCH_SIZE: model batch size according to hyper parameters
                POOL_SAMPLE_SIZE: sample extracted from X_pool during each
                                  iteration of active learning
    """
    def __init__(self, dl):
        """ initializes active learner, fitting data using parameters file """
        self.dl = dl
        logging.warning('Importing hyper parameters from file')
        assert os.path.exists('hyper_parameters.json')
        with open('hyper_parameters.json', 'r') as f:
            HYPER = json.load(f)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        if DEVICE != 'cuda':
            while True:
                ans = input('GPU not detected. Continue using CPU? [y/n] ')
                if ans == 'n': sys.exit(1)
                elif ans == 'y': break
        logging.warning('Initializing %s' % HYPER['MODEL'])
        pipeline = Pipeline([
            ('tokenizer', HuggingfacePretrainedTokenizer(HYPER['PRETRAINED_TOKENIZER'])),
            ('net', NeuralNetClassifier(
                BertModule,
                module__name=HYPER['MODEL'],
                module__num_labels=len(set(self.dl.y_train)),
                optimizer=eval(HYPER['OPTIMIZER']),
                lr=HYPER['LR'],
                max_epochs=HYPER['MAX_EPOCHS'],
                criterion=eval(HYPER['CRITERION']),
                batch_size=HYPER['BATCH_SIZE'],
                iterator_train__shuffle=True,
                device=DEVICE,
                callbacks=[
                    EarlyStopping(patience=HYPER['PATIENCE']),
                    LRScheduler(
                        LambdaLR,
                        lr_lambda=self.lr_schedule,
                        step_every='batch'),
                    ProgressBar(),
                ],
            )),
        ])
        self.num_training_steps = \
            HYPER['MAX_EPOCHS'] * (len(dl.X_train) // HYPER['BATCH_SIZE']+1)
        logging.warning('Fitting %s' % HYPER['MODEL'])
        self.model = ActiveLearner(
                estimator=pipeline,
                query_strategy=uncertainty_sampling,
                X_training=dl.X_train,
                y_training=dl.y_train
        )
        # hyper parameters that will be used elsewhere
        self.BATCH_SIZE = HYPER['BATCH_SIZE']
        self.POOL_SAMPLE_SIZE = HYPER['POOL_SAMPLE_SIZE']
        logging.warning('Model initialized')

    def lr_schedule(self, current_step):
        """ Returns the learning schedule based on training steps"""
        factor = \
                float(self.num_training_steps-current_step)/float(max(1,self.num_training_steps))
        assert factor > 0
        return factor

    def calc_score(self, X, y):
        """ Calculates the performance of the model"""
        y_pred = self.model.predict(X)
        metrics = precision_recall_fscore_support(y, y_pred, average='weighted')
        performance = {
            'precision': '{:0.2f}'.format(metrics[0]),
            'recall': '{:0.2f}'.format(metrics[1]),
            'f1': '{:0.2f}'.format(metrics[2]),
            'balanced accuracy': "%0.2f" % balanced_accuracy_score(y, y_pred)
        }
        return performance

    def query_loop(self):
        """ Active learning loop """
        accuracy_scores = \
            [float(self.calc_score(self.dl.X_dev, self.dl.y_dev)['f1'])]
        while len(self.dl.X_pool) >= self.BATCH_SIZE:
            print(f'ACCURACY\n{list(enumerate(accuracy_scores))}')
            # keeps track of annotations for current iteration in loop
            X = np.empty(self.BATCH_SIZE, dtype=object)
            Y = np.zeros(self.BATCH_SIZE, dtype=int)

            # shuffle pools of unannotated tweets
            np.random.shuffle(self.dl.X_pool)
            np.random.shuffle(self.dl.y_pool)

            # extract samples of POOL_SAMPLE_SIZE from pool
            X_pool_sample = self.dl.X_pool[:self.POOL_SAMPLE_SIZE]
            y_pool_sample = self.dl.y_pool[:self.POOL_SAMPLE_SIZE]

            # remove those samples from the pool
            self.dl.X_pool = np.delete(
                    self.dl.X_pool,
                    np.arange(self.POOL_SAMPLE_SIZE), axis=0
                    )
            self.dl.y_pool = np.delete(
                    self.dl.y_pool,
                    np.arange(self.POOL_SAMPLE_SIZE), axis=0
                    )

            for i in range(self.BATCH_SIZE):
                # query active learner
                query_idx, query_inst = self.model.query(X_pool_sample)

                # print query and get input from user regarding annotation
                print("\nANNOTATION %d/%d" % (i+1, self.BATCH_SIZE))
                print(query_inst[0])
                X[i] = query_inst[0]
                Y[i] = int(input('> '))

                # remove tweet from sample so that it won't show up again
                X_pool_sample = np.delete(X_pool_sample, query_idx, axis=0)
                y_pool_sample = np.delete(y_pool_sample, query_idx, axis=0)

            # refit model using newly annotated data
            self.model.teach(X, Y)

            # calculate new accuracy score and add to history stack
            accuracy_scores.append(
                float(self.calc_score(self.dl.X_dev, self.dl.y_dev)['f1']))

            # add unused data back to pool
            self.dl.X_pool = np.concatenate((self.dl.X_pool, X_pool_sample))
            self.dl.y_pool = np.concatenate((self.dl.y_pool, y_pool_sample))

    def predict(self, X):
        return self.model.predict(X)
