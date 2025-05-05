# Monet Style Transfer with CycleGAN

## 1. Problem Description and Data Overview

This mini-project addresses the Kaggle competition on [Monet Style Transfer using GANs](https://www.kaggle.com/competitions/gan-getting-started) where the objective is to build a generative model capable of converting landscape photographs into paintings in the style of Claude Monet. The dataset includes two unpaired image domains with approximately 1,000 Monet paintings and 7,000 landscape photos. The challenge requires using a CycleGAN architecture to perform unpaired image-to-image translation enabling generation of Monet-style images from natural photographs. The final submission is a zip archive containing generated Monet-style images corresponding to the test set photos.

## 2. Exploratory Data Analysis (EDA)

An initial exploration was conducted to inspect the data distribution, formats and resolution consistency across both domains. All images were verified to be RGB format and resized uniformly to 256×256 pixels for training consistency. Random samples from both Monet and photo domains were visualized to verify style characteristics. Image counts confirmed a class imbalance with far fewer Monet images, which was accounted for through data shuffling and repeat augmentation in the photo domain.

## 3. Data Preprocessing and Augmentation

All images were normalized to the range [-1, 1] to match the expected input scale of the CycleGAN generator and discriminator networks. Augmentation steps included random horizontal flipping and standardization. A common TensorFlow preprocessing pipeline was applied to both Monet and photo images ensuring resolution, channel consistency and performance efficiency during model training.

## 4. Model Architecture and Training Strategy

The project implemented a CycleGAN consisting of two generator networks (Monet → Photo and Photo → Monet) and two PatchGAN discriminators. The generators follow a ResNet-based architecture with residual blocks for deep feature transformation. Downsampling and upsampling were applied symmetrically to ensure shape preservation through the model. The discriminators were built as 70×70 PatchGANs allowing localized realism feedback on image patches rather than entire images.

Cycle-consistency and identity losses were used in conjunction with adversarial loss to ensure that generated images remain faithful to their original structure while acquiring stylistic characteristics. The Adam optimizer was used with a learning rate of 2e-4 and `beta_1=0.5`. Each training iteration included forward passes through all four networks and corresponding gradient updates. Training was performed for 25 epochs with careful monitoring of generated sample quality.

## 5. Results and Evaluation

During training, real-time visualization of translated images was used to track qualitative progress. Initially, outputs lacked structure and saturation, but improvements were observed across epochs as generators stabilized and captured stylistic elements of Monet’s work. The final generator produced images with recognizable painterly textures and color palettes typical of Monet paintings.

Post-training, generated Monet-style images were saved and zipped into `images.zip` following Kaggle’s naming convention. These outputs were successfully submitted to the leaderboard and achieved a MiFID score within the expected range for baseline CycleGAN models.

## 6. Discussion and Conclusion

This project successfully implemented a CycleGAN pipeline to translate natural photographs into Monet-style paintings. The model learned to emulate stylistic brush strokes and color tones with gradual improvements across training epochs. A key challenge involved balancing learning between the two domains due to dataset size disparity which was partially mitigated using augmentation and repeat sampling.

Future improvements may include:
1. Adding instance normalization in residual blocks
2. Experimenting with perceptual loss (VGG-based feature matching)
3. Testing attention-based generator architectures
4. Fine-tuning with pre-trained feature extractors for discriminator guidance

Overall however the project demonstrated the effectiveness of CycleGANs for unpaired image translation and delivered visually coherent outputs.
