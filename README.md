# CS240-Ncut-Segmentation-Efficient

Based on Tokencut(cvpr2022), uses fast NCut to replace the Original Ncut

## Motivation

- replace the Orioginal NCut method
- uses Fast Ncut [FastNcut](https://ieeexplore.ieee.org/document/5206561)

Original Implementation of NCut 

    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight

Fast Ncut

    