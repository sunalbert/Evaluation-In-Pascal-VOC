# Evaluation In Pascal-VOC

## Campute mAP in Pascal-VOC

### the input is only a dic like this.

```
detections@str = {
    classid@str: {
        imageid@numpy:[
            [xmin, ymin, xmax, ymax, confidence],
            ...
        ]
        ...
    }
    ...
}
```

### the output

this result is from the detections from the label of voc. the mAP is all 1.00

```
num_images 4952
AP for aeroplane = 1.0000
AP for bicycle = 1.0000
AP for bird = 1.0000
AP for boat = 1.0000
AP for bottle = 1.0000
AP for bus = 1.0000
AP for car = 1.0000
AP for cat = 1.0000
AP for chair = 1.0000
AP for cow = 1.0000
AP for diningtable = 1.0000
AP for dog = 1.0000
AP for horse = 1.0000
AP for motorbike = 1.0000
AP for person = 1.0000
AP for pottedplant = 1.0000
AP for sheep = 1.0000
AP for sofa = 1.0000
AP for train = 1.0000
AP for tvmonitor = 1.0000
Mean AP@0.5 = 1.0000
AP for aeroplane = 1.0000
AP for bicycle = 1.0000
AP for bird = 1.0000
AP for boat = 1.0000
AP for bottle = 1.0000
AP for bus = 1.0000
AP for car = 1.0000
AP for cat = 1.0000
AP for chair = 1.0000
AP for cow = 1.0000
AP for diningtable = 1.0000
AP for dog = 1.0000
AP for horse = 1.0000
AP for motorbike = 1.0000
AP for person = 1.0000
AP for pottedplant = 1.0000
AP for sheep = 1.0000
AP for sofa = 1.0000
AP for train = 1.0000
AP for tvmonitor = 1.0000
Mean AP@0.7 = 1.0000
```