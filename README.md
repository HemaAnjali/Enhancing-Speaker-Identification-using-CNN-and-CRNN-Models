# Enhancing-Speaker-Identification-using-CNN-and-CRNN-Models
This project uses deep learning for speaker identification with CNN, CNN+Transformer, and CNN+BiGRU models trained on the Mozilla Common Voice dataset. Performance is evaluated using sensitivity and specificity, highlighting the effectiveness of sequential models in improving speaker recognition.
# Overview
This project focuses on **speaker identification using deep learning**, evaluating three architectures: **CNN, CNN+Transformer Encoder, and CNN+BiGRU**. The models are trained on the **Mozilla Common Voice dataset** with Mel-spectrograms as input. Performance is analyzed based on **sensitivity and specificity**, highlighting the impact of sequential learning. The findings contribute to optimizing deep learning models for robust speaker recognition in real-world applications.
# Literature review
Speaker identification has evolved from traditional machine learning techniques to deep learning-based approaches for improved accuracy and robustness. Early methods relied on handcrafted feature extraction techniques such as Mel-Frequency Cepstral Coefficients (MFCCs), Perceptual Linear Prediction (PLP), and i-vectors. These features were processed using models like Gaussian Mixture Models (GMMs) and Hidden Markov Models (HMMs) for speaker classification. While effective, these models struggled with noise, variations in speech, and feature extraction complexity, limiting their real-world applications .

Deep learning has significantly enhanced speaker identification by learning feature representations directly from raw audio. Convolutional Neural Networks (CNNs) have been widely adopted due to their ability to capture spatial patterns from spectrogram representations, reducing reliance on manually crafted features. However, CNNs alone do not capture temporal dependencies in speech, leading to the adoption of hybrid architectures that integrate CNNs with sequential models. Bidirectional Gated Recurrent Units (BiGRUs) and Transformer encoders have been introduced to improve long-term feature dependencies in speaker recognition . Recent studies show that these models enhance classification performance, especially in challenging environments.

This study compares three architectures for speaker identification: Baseline CNN, CNN with Transformer Encoder, and CNN with BiGRU, focusing on their performance in terms of sensitivity and specificity rather than just accuracy. Sensitivity measures the model’s ability to correctly identify a speaker, while specificity evaluates how well it avoids false identifications. The research highlights how sequential models like BiGRU outperform pure CNNs, demonstrating better generalization and robustness in speaker classification. The findings emphasize the importance of hybrid deep learning architectures in advancing real-world speaker recognition systems.
# Architectures Used
**A)CNN (Convolutional Neural Network)**

Extracts speaker-specific features from Mel-spectrograms.

Efficient in spatial feature extraction but lacks temporal modeling.

![image](Outputs/photo_2025-02-27_16-25-44.jpg)

**B)CNN + Transformer Encoder**

Incorporates self-attention to capture global dependencies in speech.

Improves speaker identification by focusing on important time frames.
![image](Outputs/photo_2025-02-27_16-25-44.jpg)
**C)CNN + BiGRU (Bidirectional Gated Recurrent Unit)**

Enhances sequential modeling by processing speech in both forward and backward directions.

Helps in capturing contextual dependencies for better speaker discrimination.

Each architecture is evaluated based on sensitivity and specificity, ensuring robust and reliable speaker
<br>


![image](Outputs/photo_2025-02-27_16-25-30.jpg)


# Feature Importance

• Importance of individual features <br>
![image](Outputs/photo_2025-02-27_16-40-59.jpg)




# Model Evaluation



| Score | LinearRegression | Support Vector Machine | RandomForest | Gradient Boost| XGBoost|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Train Accuracy | 0.729 | -0.105 | 0.97 | 0.868 |0.870 |
| Test Accuracy | 0.806 | -0.134 | 0.882 | 0.901 | 0.904 |
| CV Score | 0.747 | 0.103 | 0.836 | 0.860 | 0.860 |

• HeatMap
![image](Outputs/photo_2025-02-27_16-25-50.jpg)

# Conclusion
Model gave 90% accuracy for Medical Insurance Amount Prediction using XGBoost. This project demonstrates the effectiveness of machine learning, particularly XGBoost, in accurately predicting medical insurance costs based on key factors. It aims to enhance cost transparency and planning, benefiting both insurers and customers.
