# DSACF

Testing and training code of paper ***HIGH SPEED AND ROBUST RGB-THERMAL TRACKING VIA DUAL ATTENTIVE STREAM SIAMESE NETWORK*** of The International Geoscience and Remote Sensing Symposium (IGARSS). 

- python3.7

- cuda11.1+pytorch1.8(not so strict)

- Ubuntu 20.04 or newer version

- install all requirements
******



# Test

Please mkdir 'dataset' at root path and download GTOT, RGB-T234 and put them into dataset directory.

```py
python test.py \
        --tools/snapshots \      #snapshot path
        --tools/result \        #results path
        --testing_dataset GTOT or RGBT234                 #testing benchmark
```

# Train
Please download LasHeR and put it into dataset directory.
```py
python train.py
```
