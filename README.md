# Image Captioning System Using Deep Learning

## Author

**Abhinaya Kuchi**  
**B.Tech – Artificial Intelligence and Data Science**

---

## 1. Project Overview

The **Image Captioning System Using Deep Learning** is an AI-based application that automatically generates meaningful natural-language descriptions for images.

This project combines **Computer Vision** and **Natural Language Processing (NLP)** using an Encoder–Decoder Deep Learning architecture.

The system uses:

- **ResNet50** as the Image Encoder
- **Image Projection Layer** for visual feature transformation
- **Word Embedding** for word representation
- **LSTM** as the Caption Decoder
- **Fully Connected Layer** for next-word prediction
- **BLEU Score** for evaluation
- **Streamlit** for the web application

### Overall Architecture

Input Image  
↓  
Image Preprocessing  
↓  
ResNet50 Encoder  
↓  
Visual Feature Extraction  
↓  
Image Projection  
↓  
Word Embedding  
↓  
LSTM Decoder  
↓  
Fully Connected Layer  
↓  
Generated Caption

---

## 2. Objectives

The main objectives of this project are:

1. To develop an automatic image captioning system using Deep Learning.
2. To extract meaningful visual features from images using ResNet50.
3. To implement an Encoder–Decoder architecture.
4. To generate captions using an LSTM-based decoder.
5. To create a vocabulary from image captions.
6. To generate descriptions for new input images.
7. To evaluate generated captions using BLEU-1, BLEU-2, BLEU-3 and BLEU-4.
8. To develop an interactive Streamlit web application.
9. To demonstrate the integration of Computer Vision and Natural Language Processing.

---

## 3. Problem Statement

Images contain rich visual information, but computers cannot directly express this information in natural language.

The objective of this project is to develop an AI system that analyzes an input image and automatically generates a meaningful textual description.

### Example

**Input:**  
An image of a dog running on grass.

**Generated Caption:**  
"A dog is running through the grass."

---

## 4. Proposed Solution

The proposed system uses an Encoder–Decoder Deep Learning architecture.

The Encoder extracts visual features from an image using a pretrained ResNet50 network.

The extracted image features are passed through an Image Projection Layer and then provided to the LSTM Decoder.

The Decoder generates the caption one word at a time.

### System Flow

Image  
↓  
ResNet50 Encoder  
↓  
Visual Features  
↓  
Image Projection  
↓  
LSTM Decoder  
↓  
Generated Caption

---

## 5. Encoder

### ResNet50

A pretrained **ResNet50** model is used as the image encoder.

The final classification layer is removed so that ResNet50 can be used as a feature extractor.

The encoder produces a **2048-dimensional visual feature vector**.

### Encoder Flow

Input Image  
↓  
Image Preprocessing  
↓  
ResNet50  
↓  
2048-D Visual Feature Vector  
↓  
Image Projection

---

## 6. Decoder

### LSTM

The decoder uses **Long Short-Term Memory (LSTM)** to generate captions sequentially.

The decoder contains:

- Word Embedding Layer
- LSTM Layer
- Fully Connected Layer

The caption generation starts with the `<start>` token and continues until the `<end>` token is generated.

### Decoder Flow

`<start>`  
↓  
Word Embedding  
↓  
LSTM  
↓  
Fully Connected Layer  
↓  
Next Word  
↓  
LSTM  
↓  
Next Word  
↓  
...  
↓  
`<end>`

---

## 7. Dataset

### Flickr8k Dataset

This project uses the **Flickr8k image-caption dataset**.

The dataset contains approximately:

- 8,000 images
- Multiple captions for each image
- More than 40,000 captions

The project contains:

- **Total captions:** 40,455
- **Vocabulary size:** 8,782 words
- **Images evaluated:** 100

The complete dataset is not included in this GitHub repository because of its size and distribution restrictions.

---

## 8. Dataset Download and Setup

Download the Flickr8k dataset from a legitimate dataset source such as Kaggle or the original Flickr8k distribution.

After downloading and extracting the dataset, create the following directory structure:

    dataset/
    ├── images/
    └── captions/
        ├── cleaned_captions.csv
        └── word_to_index.pkl

Copy the downloaded Flickr8k images into:

    dataset/images/

Place the cleaned caption file into:

    dataset/captions/cleaned_captions.csv

Place the vocabulary file into:

    dataset/captions/word_to_index.pkl

### Caption CSV Format

The CSV file should contain two columns:

    image,caption

Example:

    1000268201_693b08cb0e.jpg,A child is playing in the grass
    1000268201_693b08cb0e.jpg,A little boy is playing outside

Each image can have multiple reference captions.

---

## 9. Vocabulary

The vocabulary is created from the cleaned captions.

### Vocabulary File

    dataset/captions/word_to_index.pkl

### Vocabulary Size

    8,782 words

### Special Tokens

The model uses special tokens:

- `<start>` – indicates the beginning of a caption
- `<end>` – indicates the end of a caption
- `<pad>` – used for padding sequences
- `<unk>` – represents an unknown word

---

## 10. Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision

### Computer Vision

- ResNet50
- Pillow

### Natural Language Processing

- NLTK
- LSTM
- Word Embeddings

### Data Processing

- Pandas
- NumPy
- Pickle

### Evaluation

- BLEU Score

### Web Application

- Streamlit

---

## 11. System Requirements

Recommended environment:

- Operating System: Windows / Linux / macOS
- Python: 3.10 or higher
- RAM: 8 GB or higher recommended
- GPU: Optional

A GPU is recommended for faster training, but it is not required for running the trained model.

---

## 12. Installation

### Step 1: Clone the Repository

Run:

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd image-captioning-system

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

For Windows:

    python -m venv venv
    venv\Scripts\activate

For Linux/macOS:

    python -m venv venv
    source venv/bin/activate

### Step 3: Install Dependencies

    pip install torch torchvision pandas numpy pillow matplotlib nltk streamlit

If a `requirements.txt` file is available:

    pip install -r requirements.txt

---

## 13. Project Structure

    image-captioning-system/
    │
    ├── app.py
    ├── evaluate.py
    ├── visualize_predictions.py
    ├── plot_bleu.py
    ├── README.md
    │
    ├── models/
    │   └── best_model.pth
    │
    ├── src/
    │   └── model.py
    │
    ├── dataset/
    │   ├── images/
    │   │   └── image files
    │   │
    │   └── captions/
    │       ├── cleaned_captions.csv
    │       └── word_to_index.pkl
    │
    └── results/
        ├── prediction_1.png
        ├── prediction_2.png
        ├── prediction_3.png
        ├── prediction_4.png
        ├── prediction_5.png
        └── bleu_scores.png

---

## 14. Important Files

### app.py

Contains the Streamlit web application.

It allows users to:

- Upload an image
- Process the image
- Generate a caption
- Display the generated caption

### src/model.py

Contains the Image Captioning model architecture.

It includes:

- Image Projection Layer
- Word Embedding Layer
- LSTM
- Fully Connected Layer

### models/best_model.pth

Contains the trained model weights.

### evaluate.py

Evaluates the trained model using BLEU scores.

### visualize_predictions.py

Generates sample prediction visualizations.

### plot_bleu.py

Generates the BLEU score graph.

### cleaned_captions.csv

Contains cleaned image-caption pairs.

### word_to_index.pkl

Contains the vocabulary mapping.

---

## 15. Running the Application

Start the Streamlit application using:

    streamlit run app.py

After running the command, Streamlit will provide a local URL, usually:

    http://localhost:8501

Open the URL in a web browser.

### Important

Do not run:

    python app.py

Use:

    streamlit run app.py

---

## 16. How the Application Works

1. The user uploads an image.
2. The image is converted to RGB format.
3. ResNet50 preprocessing is applied.
4. ResNet50 extracts visual features.
5. The Image Projection Layer transforms the features.
6. The LSTM Decoder receives the image information.
7. Caption generation starts with `<start>`.
8. The LSTM predicts the next word.
9. The predicted word is passed back into the decoder.
10. The process continues until `<end>` is generated.
11. The final caption is displayed in the Streamlit application.

---

## 17. Caption Generation Algorithm

The caption generation process is:

    Load Image
        ↓
    Convert Image to RGB
        ↓
    Image Preprocessing
        ↓
    ResNet50 Feature Extraction
        ↓
    Image Projection
        ↓
    Initialize LSTM
        ↓
    Start with <start>
        ↓
    Generate Next Word
        ↓
    Feed Word Back to LSTM
        ↓
    Generate Next Word
        ↓
    Repeat
        ↓
    Stop at <end>
        ↓
    Display Generated Caption

---

## 18. Model Evaluation

The trained model is evaluated using the **BLEU (Bilingual Evaluation Understudy)** metric.

BLEU measures the similarity between generated captions and reference captions using n-gram matching.

The project evaluates:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4

Run:

    python evaluate.py

---

## 19. Evaluation Results

The trained model was evaluated on **100 images**.

### Evaluation Details

- Total Captions: 40,455
- Vocabulary Size: 8,782
- Images Evaluated: 100

### BLEU Scores

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.5264 |
| BLEU-2 | 0.3539 |
| BLEU-3 | 0.2252 |
| BLEU-4 | 0.1431 |

These results show that the model generates captions with meaningful overlap with the reference captions.

Higher-order BLEU scores are lower because matching longer sequences of words is more difficult.

---

## 20. Prediction Visualization

To generate sample prediction visualizations, run:

    python visualize_predictions.py

The generated visualizations can contain:

- Input image
- Reference captions
- Generated caption

The results are stored in:

    results/

---

## 21. BLEU Score Visualization

To generate the BLEU score graph, run:

    python plot_bleu.py

The generated graph is saved as:

    results/bleu_scores.png

---

## 22. Limitations

The generated caption may sometimes differ from the reference caption.

Possible reasons include:

- Multiple captions can correctly describe the same image.
- The model may not recognize fine-grained visual details.
- The training dataset is relatively small.
- LSTM has limitations for long sequences.
- Greedy decoding may not always produce the best caption.
- Vocabulary limitations can affect generated captions.

### Example

Reference:

    A dog is running through the grass.

Generated:

    A dog running outside.

Although the wording is different, both captions may describe the same image.

Therefore, BLEU score alone does not completely determine the quality of a generated caption.

---

## 23. Future Enhancements

Future improvements can include:

1. Attention Mechanism
2. Beam Search
3. Transformer-based Decoder
4. Larger datasets such as Flickr30k and MS COCO
5. METEOR evaluation
6. ROUGE evaluation
7. CIDEr evaluation
8. SPICE evaluation
9. Multilingual caption generation
10. Cloud deployment
11. Mobile application
12. Real-time image captioning

---

## 24. Applications

The Image Captioning System can be used for:

- Accessibility tools
- Assistive technologies
- Automatic image descriptions
- Smart photo management
- Image organization
- AI content generation
- Computer Vision applications
- Educational applications
- Human-Computer Interaction
- Image understanding systems

---

## 25. Project Achievements

The project successfully implements:

- Dataset Preparation
- Caption Preprocessing
- Vocabulary Creation
- ResNet50 Encoder
- Image Feature Extraction
- Image Projection
- Word Embedding
- LSTM Decoder
- Model Training
- Model Saving
- Caption Generation
- BLEU Evaluation
- Prediction Visualization
- BLEU Score Visualization
- Streamlit Application

---

## 26. Final Results

    Vocabulary Size : 8,782
    Total Captions  : 40,455
    Images Evaluated: 100

    BLEU-1 : 0.5264
    BLEU-2 : 0.3539
    BLEU-3 : 0.2252
    BLEU-4 : 0.1431

---

## 27. Project Status

**Completed**

The complete system performs:

    Image Input
        ↓
    Image Preprocessing
        ↓
    ResNet50 Encoder
        ↓
    Visual Feature Extraction
        ↓
    Image Projection
        ↓
    LSTM Decoder
        ↓
    Caption Generation
        ↓
    BLEU Evaluation
        ↓
    Streamlit Web Application

---

## 28. Dataset and Model Files

The Flickr8k images are not included in the GitHub repository because of their size.

The dataset should be downloaded separately and placed inside:

    dataset/images/

The cleaned captions should be placed inside:

    dataset/captions/cleaned_captions.csv

The vocabulary should be placed inside:

    dataset/captions/word_to_index.pkl

The trained model should be placed inside:

    models/best_model.pth

### Recommended .gitignore

    venv/
    __pycache__/
    *.pyc
    dataset/images/

---

## 29. Troubleshooting

### Streamlit ScriptRunContext Warning

If you see:

    missing ScriptRunContext

run:

    streamlit run app.py

instead of:

    python app.py

### Model Not Found

Make sure this file exists:

    models/best_model.pth

### Dataset Not Found

Make sure the images are located in:

    dataset/images/

### Caption File Not Found

Make sure this file exists:

    dataset/captions/cleaned_captions.csv

### Vocabulary File Not Found

Make sure this file exists:

    dataset/captions/word_to_index.pkl

---

## 30. License

This project is developed for educational and academic purposes.

The Flickr8k dataset has its own license and distribution terms. Users should download and use the dataset according to the applicable terms and conditions.

---

## 31. Acknowledgements

This project uses the following technologies and resources:

- PyTorch
- Torchvision
- ResNet50
- NLTK
- Streamlit
- Pandas
- NumPy
- Pillow
- Matplotlib
- LSTM
- Flickr8k Dataset

---

## 32. Conclusion

The **Image Captioning System Using Deep Learning** successfully combines Computer Vision and Natural Language Processing to automatically generate textual descriptions for images.

The project uses **ResNet50 as the Encoder** for visual feature extraction and **LSTM as the Decoder** for sequential caption generation.

The trained system has:

- Vocabulary Size: 8,782
- Total Captions: 40,455
- Images Evaluated: 100
- BLEU-1: 0.5264
- BLEU-2: 0.3539
- BLEU-3: 0.2252
- BLEU-4: 0.1431

The trained model is integrated with a **Streamlit web application**, allowing users to upload images and receive automatically generated captions.

This project demonstrates the practical application of:

**Deep Learning + Computer Vision + Natural Language Processing + Transfer Learning + ResNet50 + LSTM + Image Captioning + Streamlit**

---

## Author

**Abhinaya Kuchi**  
**B.Tech – Artificial Intelligence and Data Science**
