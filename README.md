# DaseAI_Final
A multimodel sentiment analysis task.

### Structure

```bash
.
├── FLAVAforMSA
│   ├── dataset
│   │   ├── data
│   │   ├── test_without_label.txt
│   │   └── train.txt
│   ├── flava
│   │   ├── __init__.py
│   │   ├── flava_example.py
│   │   ├── flava_model.py # 
│   │   ├── loader.py
│   │   └── trainer.py
│   └── run.py # Entrance of model
├── LICENSE
├── README.md
├── predict-epoch2.txt # Result for test set
├── report
│   ├── FLAVA-Structure.png
│   ├── acc.png
│   ├── f1_score.png
│   ├── loss_plot.png
│   ├── plot.py
│   ├── 多模态情感分析 实验报告.md
│   └── 多模态情感分析 实验报告.pdf # Report
├── requirements.txt
```

### Dependencies

```
numpy==1.25.2
Pillow==9.4.0
Pillow==10.2.0
Requests==2.31.0
scikit_learn==1.2.2
torch==2.0.1
torchvision==0.15.2a0
transformers==4.32.1
```

You can directly run:
```
pip install -r requirements.txt
```

### Run Model

- Run FLAVA on default setting

```py
cd ./FLAVAforMSA
```

```py
python run.py
```

You need to make your device connected to huggingface, For the first time you running, you need to download model, vocab and configurations, this will take you a while

- Customize hyperparamters for the model

```py
python run.py --epochs 10 \
							--batch_size 32 \
							--learning_rate 0.0001 \	
	            --learning_rate 0.3 \
```

- Run unimodel test

Ignore the image input: 

```
python run.py --mask_image
```

Ignore text input:

```
python run.py --mask_text
```

- Run predictions on test set

```
python run.py --predict
```

For each epoch, it will save a file like `predict-epoch2.txt`

### Attributions

No Github attributions for this repo. For FLAVA model, you can refer to:https://huggingface.co/docs/transformers/model_doc/flava