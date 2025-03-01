
# **Computer Vision Based Intruder Detection for Surveillance Cameras**  

## **Overview**  
**IntruderDetectionForSurveillanceCameras** is a **deep learning-based security application** that utilizes **Convolutional Neural Networks (CNNs) and ResNet-50** to enhance surveillance camera capabilities. This **Python-based application** can distinguish between **intruders wearing masks** and **regular civilians**, providing an **automated alert system** to notify property owners in real time.  

This project was developed to address a critical limitation of **traditional security cameras**—their inability to recognize **masked intruders** effectively. By leveraging a **dataset of 13,500 images**, including individuals with **facial coverings** and **regular headshots**, the model is trained to **detect and classify potential threats** accurately.  

To account for **winter conditions**, where non-intruders may be wearing masks for warmth, the system incorporates a **5-second delay** to allow individuals to **reveal their face** before triggering an alert.  

---

## **Key Features**  
✔ **Facial Detection & Intruder Classification** – Uses **CNNs with ResNet-50** to classify individuals as **civilians or intruders wearing masks**.  
✔ **Large-Scale Dataset** – Trained on **13,500 images**, including various facial coverings for robust detection.  
✔ **Real-Time Monitoring** – Integrates with security cameras to **continuously analyze footage** and detect suspicious activity.  
✔ **Automated Alerts** – Notifies the owner **immediately** when an **intruder is detected**, allowing for quick action.  
✔ **Winter Mode Delay** – Introduces a **5-second buffer period** to prevent false alarms from **non-intruders wearing masks for warmth**.  

---

## **Technical Details**  

### **1️⃣ Deep Learning Model: ResNet-50**  
This project implements **ResNet-50**, a powerful **Convolutional Neural Network (CNN)** known for its ability to recognize patterns in images. The model is **pre-trained on ImageNet** and further fine-tuned using our **intruder dataset** for specialized facial recognition tasks.  

**Model Architecture:**  
- **Convolutional layers** extract facial features.  
- **Residual connections** improve gradient flow and prevent vanishing gradients.  
- **Fully connected layers** classify individuals as **civilians or masked intruders**.  

---

### **2️⃣ Dataset**  
The model is trained on a **custom dataset of 13,500 labeled images**, including:  
📌 **Masked Intruders** – Individuals wearing facial coverings such as **balaclavas, ski masks, and surgical masks**.  
📌 **Civilians** – Individuals with fully visible faces in **varied lighting and angles**.  

The dataset ensures the model can handle **different environments, facial coverings, and occlusions** for **real-world accuracy**.  

---

### **3️⃣ Implementation Steps**  

#### **🔹 Data Preprocessing**
- **Image resizing** to match ResNet-50 input dimensions.  
- **Normalization** for consistent pixel values.  
- **Augmentation** (rotation, flipping, brightness adjustments) to improve model generalization.  

#### **🔹 Model Training & Fine-Tuning**
- Fine-tuning **ResNet-50** with our **intruder dataset**.  
- Using **transfer learning** for enhanced accuracy.  
- Optimizing with **Adam optimizer & Cross-Entropy loss function**.  

#### **🔹 Real-Time Detection Pipeline**
1. Capture live video feed from **surveillance cameras**.  
2. Perform **face detection & classification** using the trained model.  
3. If an **intruder is detected**, wait **5 seconds** (winter mode).  
4. If the face remains **covered**, **trigger an alert** to the owner.  

---


## **Expected Output**  
📌 **Intruder Detected (Alert Triggered)** – If a masked face is detected beyond **5 seconds**, an **alert notification** is sent.  
📌 **Civilians Identified (No Alert)** – If a face is uncovered within **5 seconds**, the system dismisses the alert.  
📌 **Live Monitoring Dashboard** – Displays **real-time facial recognition and classification results**.  

---

## **Results & Accuracy**  
- **Trained for 30+ epochs**, achieving **high accuracy (>90%)** on test data.  
- **Effectively detects masked individuals** while **minimizing false positives** in **winter conditions**.  
- **Outperforms traditional security cameras** by enabling **intruder detection even with face coverings**.  

---

## **Future Enhancements**  
🔹 **Additional Object Detection (Weapons, Knives, Break-In Devices** – The model could be trained on an even larger dataset that includes arms and break-in tools to notify the user. 

🔹 **Integration with Smart Home Systems** – Sync alerts with **smart locks, alarms, and security apps**.  
🔹 **Expanded Dataset** – Incorporate **thermal imaging** for night-time intruder detection.  
🔹 **Multi-Camera Support** – Enable simultaneous monitoring across multiple surveillance feeds.  
🔹 **Cloud-Based Processing** – Improve speed and efficiency using **edge AI and cloud computing**.  

---

## **References**  
- **ResNet-50 Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
- **Face Detection with OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)  
- **TensorFlow & Keras**: [Official Guide](https://www.tensorflow.org/tutorials)  

---

