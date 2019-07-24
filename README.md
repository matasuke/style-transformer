# Style transformer experimental modified

Experimental environment for style-transformer

## requirements
- python 3.7
  - pytorch 1.1.0+
  - torchtext 0.3.1*+
  - fastText
  - kenlm

- install all requirements by pipenv
```
pipenv install --dev
```

- kenlm
  you must set path for lmplz to be accessible by non-root user like below this
  ```
  ln -s ~/usr/local/bin/ /ahc/mlocal/mosesdecoder/mosesdecoder-RELEASE-4.0/bin/lmplz
  ```

## usage
first, you should enter pipenv environment
```
pipenv shell
```

### with default config (YELP dataset)
```
python main.py -c data/config/default.json -d 0
```

- arguments
  - -c, --config: path to configuration file
  - -d --device: gpu id to use

### configration file
```
{
    "name": "yelp",  // training session name
    "n_gpu": 1, // the number of GPUs to use for training
    "data": {
        "data_root": "./data/corpora/yelp/",  # data root path
        "train_pos": "train.pos",  # file name of positive train corpora, which is pre-tokenized
        "train_neg": "train.neg",  # file name of negative train corpora, which is pre-tokenized
        "test_pos": "test.pos",  # file name of positive test corpora, which is pre-tokenized
        "test_neg" :"test.neg",  # file name of negative test corpora, which is pre-tokenized
        "min_freq": 3,  # minumum frequency which is used for vocabulary
        "batch_size": 64,  # batch size
        "embed_size": 256,  # embed size for vocaburaly and style embedding
        "shuffle": "true",  # shuffle train data
        "load_pretrained_embed": false,  # train vocaburaly emvedding before training if true
        "pretrained_embed_path": "embedding/"  # path to save pretrained embedding
    },
    "evaluator": {
        "args": {
            "text_paths": [
                "data/corpora/yelp/train.pos",  # path to positive corpora, which is same with train_pos
                "data/corpora/yelp/train.neg"  # path to negative corpora, which is same with train_neg
            ],
            "style_list": ["positive", "negative"],  # style name corresponding to text_paths
            "reference_paths": [
                "data/corpora/yelp/reference.pos",  # path to positive reference if exists
                "data/corpora/yelp/reference.neg"  # path to negative reference if exists
            ],
            "lm_config": { # configuration for language model
                "n_order": 5  # n-gram
            },
            "classifier_config": {  # configuration for style classifier
                "epoch": 25,  # total epoch to train classifier
                "lr": 1.0,  # learning rate
                "n_order": 2,  # n-gram
                "verbose": 2,  # show log
                "min_count": 1,  # minumum frequency to count words
                "label_prefix": "__style__"  # label prefix
            }
        },
        "save_path": null  # this must be null when it's not resuming training
    },
    "arch_generator": {
        "type": "StyleTransformer",  # name of generator architecture
        "args": {
             "num_styles": 2,  # the number of styles
             "num_layers": 4,  # the number of layers
             "num_heads": 4,  # the number of heads in multi-head attention
             "hidden_dim": 256,  # size of hidden dimension
             "max_seq_len": 25,  # max sequence length
             "dropout": 0.0,  # dropout ratio
             "load_pretrained_embed": false   # load pretrained embedding
        }
    },
    "arch_discriminator": {
        "type": "Discriminator",  # name of discriminator architecture
        "args": {
             "num_styles": 2,  # the number of styles
             "num_layers": 4,  # the numb er of layers
             "num_heads": 4,  # the number of heads in multi-head attention
             "hidden_dim": 256,  # size of hidden dimenstion
             "max_seq_len": 25,  # max sequence length
             "dropout": 0.0,  # dropout ratio
             "discriminator_method": "Multi",  # discriminator method "Multi" or "Conditional"
             "load_pretrained_embed": false  # load pretrained embedding
        }
    },
    "generator_optimizer": {
        "type": "Optim",  # name of optimizer module
        "args":{
            "method":"adam",  # optimizer type
            "lr": 0.0001,  # learning rate
            "max_grad_norm": 5,  # grad norm
            "lr_scheduler": null,  # learning rate scheduler
            "weight_decay": 0.0,  # weight decay
            "lr_decay": 1.0,  # decay rate
            "start_decay_at": null  # start decay at specified iteration
        }
    },
    "discriminator_optimizer": {
        "type": "Optim",  # name of optimizer module
        "args": {
            "method": "adam",  # optimizer type
            "lr": 0.0001,  # learning rate
            "max_grad_norm": 5,  # grad norm
            "lr_scheduler": null,  # learning rate scheduler
            "weight_decay": 0.0,  # weight decay
            "lr_decay": 1.0,  # decay rate
            "start_decay_at": null  # start decay at specified iteration
        }
    },
    "trainer": {
        "pretrain_generator_steps": 500,  # the number of iterations to pre-train generator
        "generator_steps" : 5,  # steps to train generator continuously
        "discriminator_steps": 10,  # steps to train discriminator continuously
        "save_dir": "data/saved/",  # save directory
        "evaluation_step": 25,  # evaluation step
        "log_step": 5,  # logging step
        "tensorboardX": true,
        "log_dir": "data/saved/runs",
        "drop_rate_config": [[1, 0]],
        "temperature_config": [[1, 0]],
        "word_drop_prob": 0,
        "split_pretrain_data": false,  # feed positive and negative corpora separately when pretraining
        "split_train_data": false,  # feed positive and negative corpora separately when pretraining
        "pos_update_rate": {  # these are used when split_train_data is True
            "self_factor": null,  # update rate for positive self-reconstruction loss
            "cycle_factor": null,  # update rate for positive cycle loss
            "adv_factor": null  # update rate for positive adversarial loss
        },
        "neg_update_rate": {  # these are used when split_train_data is true
            "self_factor": null,  # update rate for negative self-reconstruction loss
            "cycle_factor": null,  # update rate for negative cycle loss
            "adv_factor": null  # update rate for negative adversarial loss
        },
        "factors": {  # factores multiplied for loss
            "self_factor": 0.25,  # factor for self-reconstruction loss
            "cycle_factor": 0.5,  # factor for cycle loss
            "adv_factor": 1.0,  # factor for adversarial loss
            # factores below are used only when split_train_data is true
            "pos_self_factor": null,  # factor for positive self-reconstruction loss
            "neg_self_factor": null,  # factor for negative self-reconstuction loss
            "pos_cycle_factor": null,  # factor for positive cycle loss
            "neg_cycle_factor": null,  # factor for negative cycle loss
            "pos_adv_factor": null,  # factor for positive adversarial loss
            "neg_adv_factor": null  # factor for negative adversarial loss
        }
    }
}

```
