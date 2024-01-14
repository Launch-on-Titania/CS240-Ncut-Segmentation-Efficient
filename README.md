# CS240-TokenCut(NCut)-Segmentation-Efficient

Based on TokenCut(CVPR2022), uses fast NCut to replace the Original NCut

## Motivation

- Replace the Original NCut method
- Uses Fast NCut [FastNCut](https://ieeexplore.ieee.org/document/5206561)

Original Implementation of NCut

    Implementation of NCut Method.
    Inputs
      feats: the pixel/patches features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: threshold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight

Fast NCut
    Achieved on TokenCut/fast_ncut.py

## Usage
    
``` 
# single image
    python main_tokencut.py --image_path examples/VOC07_000012.jpg --visualize all

# dataset
     python main_tokencut.py --dataset COCO20k --set train

```