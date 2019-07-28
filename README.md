### **“达观杯”文本智能信息抽取挑战赛**
[比赛链接](https://biendata.com/competition/datagrand/)  
官方提供了基于CRF++的baseline，训练很快，线上测试为0.85083  
CRF++的模型预测官方提供的corpus.txt，再预测test.txt，效果会好那么一点  
  
基于开源[LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)，主要对train.py和predictor.py做了修改，线上测试0.842887  
很是尴尬，没超过baseline  
应该是参数没调好，参数大小都设置的比较小，也没加词向量  
  
* pre_data.py 是生成训练、验证文件和待预测的测试文件、提交文件，代码很简单  
* train.py 训练下改下pre_data.py生成文件的路径  
* predict.py 生成输出文件，和baseline模型输出文件一样  

可以加[glove](https://github.com/stanfordnlp/GloVe)、[fasttext](https://github.com/facebookresearch/fastText)等生成的词向量  

:blush: 整理有点粗忙，可能存在bug，欢迎指出，一起进步
