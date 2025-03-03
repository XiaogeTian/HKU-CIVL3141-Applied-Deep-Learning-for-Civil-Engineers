# HKU-CIVL3141-Applied-Deep-Learning-for-Civil-Engineers
Course instructor: Prof. Jiaji Wang and Prof. Xugang Wang.
For quesitons related to this repo, please contact corresponding TA Mr Tian Xiaoge (Email: xiaogetian@connect.hku.hk) or use Ed forum
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



# CIVL3141 Tutorials

Below are the tutorials for CIVL3141, organized by topic. Click the badges to open in Google Colab.

| Tutorial Topic             | Tutorial Number | Link                                                                 |
|----------------------------|-----------------|----------------------------------------------------------------------|
| **GAN**                   | Tutorial 1      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Egr0YYKu2bq6xNfPHOepdMI3YXPHVjgH) |
|                            | Tutorial 2      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Y0JCEw4OTqmRzndjkxtTsSZhjJOzUzSh) |
|                            | Tutorial 3      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iHfaqpv1yTwF_jGO9M47HdmU0cW6bTxV) |
| **Diffusion**             | Tutorial 1      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AygAtKEJi5sl-w4wb65nCbbzvS59c0pr) |
|                            | Tutorial 2      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SpU_J6XLrBvjnWj0CSgwl5JHZuhwAhfs) |
|                            | Tutorial 3      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/112gMKP0nGR9rGR_Z1XwFLmRYir1GxQos) |
| **Transformer**           | Tutorial 1      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Uq247eb4wlVQx5EpbFM1Zcz7nQ4Z669N) |
| **LLM-GPT API Call**      | Tutorial 1      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p0WsflZte0sQExjj1RRZODZU6Mv8_Ktl) |
|                            | Tutorial 2      | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Y8fe8HxPXOB4AlvzbV-jlDzAr68ku4NH) |

**Note**: Colab is free to use with T4 GPU. Students can explore how to use Colab with the links below:
- [![YouTube](https://img.shields.io/badge/YouTube-Watch-red)](https://www.youtube.com/watch?v=Ii6gs9zADEA)  
- [![Bilibili](https://img.shields.io/badge/Bilibili-Watch-blue)](https://www.bilibili.com/video/BV13K4y1P7dx/?spm_id_from=333.337.search-card.all.click&vd_source=50c2bd940a07d78c3da3454e340a992f)

For the LLM-GPT api call part, Qwen model (通义千问百炼大模型) is free to use within 6 months with free capacity of 500k tokens. Please visit their official website to sign up and get your own api-key to use in the tutorial notebook. 
- If you have alipay account:https://www.alibabacloud.com/en/solutions/generative-ai/qwen?_p_lc=1
- If you want to use email: https://www.alibabacloud.com/en/solutions/generative-ai/qwen?_p_lc=1
- There are also tutorials about the Qwen model api call on videos. Please check below:
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red)](https://www.youtube.com/watch?v=wijvCBmzVvE)  

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
- TensorFlow, G. (2015). Large-scale machine learning on heterogeneous systems. Google Research, 10, s15326985ep4001.

- Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., & Lerer, A. (2017). Automatic Differentiation in PyTorch. In *Proceedings of the 31st Conference on Neural Information Processing Systems (NIPS)*.

- Neuromatch Academy. (n.d.). Introduction to Deep Learning with PyTorch. Retrieved from https://github.com/NeuromatchAcademy/course-content-dl. Licensed under Creative Commons Attribution 4.0 International License (CC BY 4.0) and BSD (3-Clause) License.

- Inkawhich, N. A., Wen, W., Li, H., & Chen, Y. (2019). Feature Space Perturbations Yield More Transferable Adversarial Examples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. 

- Jupyter Development Team. (2023). Jupyter Notebook. Retrieved from https://jupyter.org/

- Inkawhich, N. A. (2023). GitHub - inkawhich. Retrieved from https://github.com/inkawhich 
