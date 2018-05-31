# Evaluation In Pascal-VOC

## Campute mAP in Pascal-VOC

### the input is only a 2D list like this.

```
detection[classid@int(0~21)][imageind@int(0~4952)] = [[xmin, ymin, xmax, ymax, confidence],]
detections@dic = [
    classid@int: [
        imageind@int:[
            [xmin, ymin, xmax, ymax, confidence],
            ...
        ]@numpy
        ...
    ]
    ...
]
```

### note

imageind is the index in test.txt

### the output

this result is from the detections from the label of voc. the mAP is all 1.00

```
num_images 4952
AP for aeroplane = 0.4179
AP for bicycle = 0.3738
AP for bird = 0.3732
AP for boat = 0.2918
AP for bottle = 0.3380
AP for bus = 0.3473
AP for car = 0.3981
AP for cat = 0.4366
AP for chair = 0.3846
AP for cow = 0.3964
AP for diningtable = 0.0519
AP for dog = 0.4226
AP for horse = 0.4957
AP for motorbike = 0.3655
AP for person = 0.4287
AP for pottedplant = 0.2134
AP for sheep = 0.4406
AP for sofa = 0.2778
AP for train = 0.2964
AP for tvmonitor = 0.4759
Mean AP@0.5 = 0.3613
AP for aeroplane = 0.0578
AP for bicycle = 0.0147
AP for bird = 0.0199
AP for boat = 0.0950
AP for bottle = 0.0186
AP for bus = 0.0218
AP for car = 0.0249
AP for cat = 0.0385
AP for chair = 0.0445
AP for cow = 0.0383
AP for diningtable = 0.0006
AP for dog = 0.0714
AP for horse = 0.0233
AP for motorbike = 0.0255
AP for person = 0.0640
AP for pottedplant = 0.0176
AP for sheep = 0.0246
AP for sofa = 0.0187
AP for train = 0.0398
AP for tvmonitor = 0.1221
Mean AP@0.7 = 0.0391
```
