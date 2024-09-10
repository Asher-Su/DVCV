# DVCV: Dynamic Voting Cross Validation Based Federated Learning Scheme Against Untargeted Poisoning Attack

## Abstract
The underlying architecture of federated learning potentially makes it vulnerable to poisoning attacks. Many detection methods have successfully identified the attacker, but they are struggling to maintain the detection accuracy and target model’s performance in different scenarios, especially facing to non-IID scenarios. In this paper, we propose a novel Dynamic Voting Cross-Validation (DVCV) method for untargeted poisoning attack detection in non-IID federated learning. To accurately identify suspicious clients, we first design a class weight voting mechanism to maintenance a voting list through verifying the accuracy of the global model with the current sub-model in each client. Then, we employ a dynamic voting scheme to adaptively scale weights updation according to the obtained voting list. Extensive experiments on two well-known datasets demonstrate DVCV outperforms state-of-the-art defense method under five advanced untargeted attacks both in IID and non-IID scenarios, while preserving the target model’s performance. 

## 1. Required Packages
* pytorch = 2.2.2 <br>
* torchvision = 0.17.2 <br>
* tqdm = 4.62.3 <br>
* matplotlib = 3.8.4 <br>
* scipy = 1.11.4 <br>

## 2. Document structure
**args.py ---> Parameter Setting** <br>
**datasets.py ---> Dataset** <br>
**defense.py ---> Our Defense And Zhao's Defense** <br>
**poison.py ---> Five Sota Untargeted Attacks** <br>
**utils.py ---> Tool Functions** <br>
