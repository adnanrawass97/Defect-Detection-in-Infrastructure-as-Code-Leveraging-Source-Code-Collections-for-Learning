# Defect Detection in Infrastructure-as-Code: Leveraging Source Code Collections for Learning

This project uses **Artificial Intelligence (AI)** to detect **misconfigurations** that can lead to **vulnerabilities** in Infrastructure-as-Code (IaC) scripts. Our goal is to ensure **secure deployments** for tools like **Puppet** and **Ansible** by providing smart, automated insights tailored for **DevOps** practices.

This project replicates and extends two major works using a **deep learning** approach to compare with traditional **machine learning** techniques:

### 1. Characterizing Defective Configuration Code Scripts Used For Continuous Deployment by Akond Rahman
   - **Focus**: Defective Puppet scripts.
   - **Technique**: Text mining using **TF-IDF** and **Bag-of-Words** for feature extraction, classified using **Random Forest**.
   - **GitHub Repository**: [Akond's Puppet Work](https://github.com/akondrahman/IaCExtraction/tree/master/ist2018_src)
   - **Dataset**: [Characterizing Defective Configuration Code Scripts Dataset](https://figshare.com/articles/dataset/Characterizing_Defective_Configuration_Code_Scripts_Used_For_Continuous_Deployment/5729535)

### 2. Within-Project Defect Prediction of Infrastructure-as-Code Using Product and Process Metrics by Stefano Dalla Palma
   - **Focus**: Defective Ansible scripts.
   - **Technique**: Text mining with **Bag-of-Words** for feature extraction, classified using **Random Forest**.
   - **GitHub Repository**: [Stefano's Ansible Work](https://github.com/stefanodallapalma/TSE-2020-05-0217/tree/master?tab=readme-ov-file)
   - **Dataset**: [Defect Prediction in IaC Scripts Dataset](https://zenodo.org/records/4299908)

## Our Contribution
We extend and replicate these works by:
1. **Using Deep Learning Models**: We compare the traditional **machine learning** approaches (TF-IDF, Bag-of-Words, and Random Forest) used in both studies with **deep learning models**:
   - **Longformer**: A transformer-based model that processes long sequences, making it suitable for long IaC scripts.
   - **CodeBERT**: A language model specifically designed to understand code, offering a more nuanced approach to code syntax and semantics.

2. **Comparison of Approaches**:
   - **Text Mining + Random Forest** (Used in both original works) vs. **Deep Learning** (Our approach).
   - We aim to demonstrate how modern **deep learning** models can outperform traditional machine learning techniques in detecting misconfigurations in IaC scripts.

## Methodology
- **Feature Extraction**: We employ both **TF-IDF**, **Bag-of-Words**, and deep learning models to extract features from IaC scripts (Puppet and Ansible).
- **Classification**: We compare the performance of:
   - Traditional **Random Forest** classifiers.
   - Deep learning models (**Longformer** and **CodeBERT**), which understand the context of IaC scripts at a deeper level.

## Results and Insights
Our experiments compare the accuracy, precision, recall, and F1-score of the machine learning and deep learning approaches. By leveraging **deep learning**, we aim to:
- Improve detection accuracy of **defective configurations**.
- Provide more robust insights into how **IaC scripts** can be made secure through automated analysis.

## Conclusion
This project highlights the effectiveness of using **deep learning** in detecting defects in **IaC** scripts compared to traditional **machine learning** approaches. By combining insights from previous research and modern techniques, we aim to enhance the security of DevOps pipelines by automating defect detection in IaC configurations.

---

### References:
1. Rahman, A., & Williams, L. **Characterizing Defective Configuration Code Scripts Used For Continuous Deployment**. [GitHub Repository](https://github.com/akondrahman/IaCExtraction/tree/master/ist2018_src), [Dataset](https://figshare.com/articles/dataset/Characterizing_Defective_Configuration_Code_Scripts_Used_For_Continuous_Deployment/5729535).
   - This work used both **TF-IDF** and **Bag-of-Words** for feature extraction.
   
2. Dalla Palma, S., & Lambiase, G. **Within-Project Defect Prediction of Infrastructure-as-Code Using Product and Process Metrics**. [GitHub Repository](https://github.com/stefanodallapalma/TSE-2020-05-0217/tree/master?tab=readme-ov-file), [Dataset](https://zenodo.org/records/4299908).
