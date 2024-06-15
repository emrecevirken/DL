import tensorflow as tf
import os
import numpy as np
from tensorflow.python.keras import models , layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = 256
BATCH_SIZE = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "./PlantVillage",
    shuffle=True, #görüntülerin karşılaştırılması
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE #modelin her iter. işlediği görüntü sayı.
)
#sınıf isimleri ve sayılarının alınıp yazdırılması
class_names = dataset.class_names
print(class_names)
n_classes = len(class_names)

loaded_model = tf.keras.models.load_model('.\models\my_model.keras')


#tahmin fonksiyonunun tanımlanması
def predict_image_class_and_accuracy(img_path, class_names): #Bir görüntünün sınıfını ve tahmin doğruluğunu belirler.
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)) #gör. yüklenmesi ve hedef boyuta yeniden boyutlandırma
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) #gör. dizisinin genişletilmesi

    predictions = loaded_model.predict(img_array) #modelin kul. tahmin yapılması
    predicted_class = np.argmax(predictions[0]) #en yüksek olasılık sahıp sınıf index belirlenmesi
    confidence = np.max(predictions[0]) #tahmin dizisindeki en yüksek olasılığın alınması
    
    
    predicted_class_name = class_names[predicted_class]
    
    return predicted_class_name, confidence

test_folder_path = "./test"

test_images = [os.path.join(test_folder_path, img) for img in os.listdir(test_folder_path) if img.endswith(".JPG")]
x=0
for test_image_path in test_images:
    x+=1
    #predicted_class, confidence = predict_image_class_and_accuracy(loaded_model, test_image_path, class_names)
    predicted_class, confidence = predict_image_class_and_accuracy(test_image_path, class_names) #Güven değeri = görüntünün sınıfını, tahmin doğruluğu.
    print(f'Test Fotoğrafı: {os.path.basename(test_image_path)}')
    print(f'Tahmin edilen sınıf: {predicted_class}')
    print(f'Doğruluk oranı: {confidence * 100:.2f}%')
    print("-" * 50)
    
print(x)
