# DeepCIN
Feature based Sequential Classifier with Attention Mechanism

@author: Sudhir Sornapudi
@email: ssbw5@mst.edu

Arxiv paper: https://arxiv.org/abs/2007.11392

Step I. Localization
- Execute 'data_gen' folder codes
	- [MATLAB] Generate vertical segments [Run 'main_getSegmentedImages.m']
	- [Python] Preprocess the vertical images to reshape images to size 64x704
	- Saves images in separate folder
	
Step II. Segment-level Sequence Generator
- Run 'main_seg_level_sequence_gen.py'
- Reads vertical segment images and csv containing ground truths
- The data is split at image-level for individual folds and the class distribution is maintained (Stratified K-fold)
- For each fold, saves the logit vector data and trained model weights

Step III. Image-level Classifier
- Run 'main_attention_based_fusion.py'
- For each fold, loads the logit vector data and saves trained model weights 
