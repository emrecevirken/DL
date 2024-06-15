import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)

# Yüklemeler için bir dizin belirliyoruz
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model dosyası
model_path = '.\models\model.h5'

# Keras modelini yüklüyoruz
model = load_model(model_path)
print('Model yüklendi. Check http://127.0.0.1:5000/')

# Sınıf etiketlerini tanımlıyoruz
labels = {0: 'Sağlıklı', 1: 'Tozlu', 2: 'Hastalıklı'}

# Görüntüden tahmin yapan fonksiyon
def getResult(image_path):
    # Görüntüyü belirtilen boyutlarda yüklüyoruz
    img = load_img(image_path, target_size=(225, 225))
    # Görüntüyü numpy array formatına dönüştürüyoruz
    x = img_to_array(img)
    # Veriyi 0-1 aralığında normalleştiriyoruz
    x = x.astype('float32') / 255.
    # Modelin girdi olarak alabilmesi için boyutunu genişletiyoruz
    x = np.expand_dims(x, axis=0)
    # Model ile tahmin yapıyoruz
    predictions = model.predict(x)[0]
    # Tahminleri döndürüyoruz
    return predictions

# Ana sayfa rotasını tanımlıyoruz
@app.route('/')
def home():
    return render_template('index.html')

# Görüntü yükleme ve tahmin rotasını tanımlıyoruz
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Dosya bulunamadı'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'Dosya adı boş'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        predictions = getResult(file_path)
        label_index = np.argmax(predictions)
        label = labels[label_index]
        confidence = predictions[label_index]
        
        return f'Tahmin: {label}, Güven: {confidence:.2f}'
    
    return 'Bir hata oluştu'

# Uygulamayı başlatıyoruz
if __name__ == '__main__':
    # Yüklemeler için klasör yoksa oluşturuyoruz
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
