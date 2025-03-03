# HKU-CIVL3141-Applied-Deep-Learning-for-Civil-Engineers
Course instructor: Dr. Jiaji Wang(cewang@hku.hk) and Dr. Xugang Wang(xuguangw@hku.hk)  
## Course Description
This course is a bachelor year-four level tutorial that introduces the theory and application of Generative AI across various domains, including computer generative AI in conducting natural language processing, images generation, and Large Language Model api call. The course will first illustrate existing examples and applications of Generative AI, helping students understand the foundational concepts and methodologies. Then, the course will delve into advanced topics, including **Generative Adversarial Networks (GANs), Diffusion model, Transformer, and transformer-based Large Language Model api call**. In the meantime, the course will demonstrate the application of Generative AI in solving real-world problems, such as image generation, text generation, and data augmentation.


Upon completion of this course, students are expected to be able to conduct the following:
- Formulating real-world applications into Generative AI problems and identify the related learning issues;
- Selecting and applying the most suitable methods to solve specific problems;
- Comparing different Generative AI approaches based on common performance criteria.
## Prerequisites
- Students are expected to have the following background:
- Basic knowledge in computer science and programming;
- Knowledge of basic machine learning principles. However, it is also ok if you are not familiar with deep learning. We will offer lectures and tutorials to teach you how to implement Generative AI models with Python.


# Colab notebook
| Tutorial            | Link                                                                 |
|---------------------|----------------------------------------------------------------------|
| CIVL3141 GAN Tutorial 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 GAN Tutorial 2 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 GAN Tutorial 3 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 Diffusion Tutorial 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 Diffusion Tutorial 2 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 Diffusion Tutorial 3 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 Transformer Tutorial | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 LLM-GPT API Call Tutorial 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
| CIVL3141 LLM-GPT API Call Tutorial 1 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |


# GPU requirements
- Students with GPU resources can try to create the environment by themselves. If students don't have available GPU to use, you can visit the colab folder where the same repo are uploaded. In colab, you don't have to hold a GPU to run the codes. You can use the default T4 GPU to use. 
- Colab link：




## Reading Materials
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville, 2016. Deep Learning. MIT Press.
- Aaron Courville and Ian Goodfellow, 2014. Generative Adversarial Networks. arXiv:1406.2661.
- Diederik P. Kingma and Max Welling, 2013. Auto-Encoding Variational Bayes. arXiv:1312.6114.
- Jayesh K. Gupta, 2020. Generative Adversarial Networks in Action. Manning Publications.




## Installing programing packages
- During installing the packages, it is recommended to use Anaconda as the environment setup tool for creating the virtual environment. 
Please check the tutorial about how to use Anaconda to build up an environment for establishing deep learning framework Pytorch/Tensorflow (Pytorch will be the one this repo utilized). 

- Python
  - Python 3.8
    Installation of a Python distribution such as Anaconda recommended
    Quick start with LearnPython.org (<url id="cusrgfj1huinrmadcb2g" type="url" status="parsed" title="Learn Python - Free Interactive Python Tutorial" wc="1632">http://learnpython.org/</url> ) and DataCamp (<url id="cusrgfj1huinrmadcb30" type="url" status="parsed" title="Just a moment..." wc="159">http://www.datacamp.com/courses/tech:python</url> )

- Anaconda installation tutorial: https://www.youtube.com/watch?v=5mDYijMfSzs
- Torch installation tutorial: https://www.youtube.com/watch?v=EMXfZB8FVUA
- Using **"pip install -r requirement.txt"** to install the required packages. Please adjust the version of **pytorch and cuda** based on your **gpu certain type**.

- PyTorch website
  - Installation from <url id="cusrgfj1huinrmadcb4g" type="url" status="parsed" title="PyTorch" wc="2468">https://pytorch.org/</url>



## Reference
- Neuromatch Academy. (n.d.). Introduction to Deep Learning with PyTorch. Retrieved from https://github.com/NeuromatchAcademy/course-content-dl. Licensed under Creative Commons Attribution 4.0 International License (CC BY 4.0) and BSD (3-Clause) License.

- Inkawhich, N. A., Wen, W., Li, H., & Chen, Y. (2019). Feature Space Perturbations Yield More Transferable Adversarial Examples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. 

- Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., & Lerer, A. (2017). Automatic Differentiation in PyTorch. In *Proceedings of the 31st Conference on Neural Information Processing Systems (NIPS)*.

- Jupyter Development Team. (2023). Jupyter Notebook. Retrieved from https://jupyter.org/

- Inkawhich, N. A. (2023). GitHub - inkawhich. Retrieved from https://github.com/inkawhich 
