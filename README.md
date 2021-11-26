# dupimage
  Code from the paper "Deduplication of Retrieved Image Data Using Deep Network Features".

## Abstract
  In this paper, we propose a method for duplicating retrieved image data for image recommendation system. From the candidate images that are retrieved for a given queries, the proposed system finds duplicate images with no cognitive difference and make them into one cluster. By choosing a representative image from each clusters it is possible to avoid useless memory consumption and give users more compact retrieval result. The proposed system is in the form of two modules connected. First, the feature extraction module represents candidate images as cognitive features obtained by using a pre-trained deep network model. Second, hierarchical clustering is applied to the feature vectors in order to find clusters with duplicated images. Through computational experiments, we confirm that the proposed method has competitive performance compared with a well-designed image processing module: using several handcrafted filters.

## Acknowledgement
  This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No. 2016-0-00145, Smart Summary Report Generation from Big Data Related to a Topic)

## Copyright
  This software is not commercially available, and the copyright belongs to Kyungpook National University Industry-Academic Cooperation Foundation.
