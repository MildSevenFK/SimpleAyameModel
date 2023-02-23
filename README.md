# SimpleAyameModel
A simple learning model using pytorch to distinguish between Nakiri Ayame and other characters.



![isAyame](https://user-images.githubusercontent.com/125388076/220857980-9ac002da-6e7a-40b0-9d9a-4aa80f3b9d12.gif)


This project is a machine learning model based on PyTorch that can classify Nakiri Ayame, a popular virtual YouTuber (VTuber), and other characters. The model was trained using PyTorch and can be accessed via the Hugging Face website's demo page [here](https://huggingface.co/spaces/MildSevenFK/AyameModel). 

The repository contains the following files:


>1.AyameModelPredict.py:　A script that uses the trained model to make predictions<br>
>2.AyameModelTrain.py:　A script used to train the model<br>
>3.ayame_model_1.pt:　The trained model<br>
>4.app.py:　A demo app that showcases the model's capabilities<br>

### Usage

To use the model, simply access the demo page hosted on Hugging Face's website here and follow the instructions provided. Alternatively, you can use the AyameModelPredict.py script to make predictions using the model.

To train your own model, you can use the AyameModelTrain.py script. However, this requires a dataset of images labeled with the character names you wish to classify.

### Requirements

PyTorch
Torchvision
Flask
Flask-Cors
These requirements can be installed via pip using the following command:

```
pip install -r requirements.txt
```

Acknowledgements
This project was made possible by the PyTorch framework and the Hugging Face website, which provided access to the demo page and hosted the trained model.
