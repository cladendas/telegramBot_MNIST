import telebot
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from google.colab import files
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
%matplotlib inline

model = Sequential() #Создаём сеть прямого распространения
model.add(Dense(5000, input_dim=784, activation='relu')) #Добавляем полносвязный слой на 800 нейронов с relu-активацией (активаци при значении >= 0), input_dim - кол-во элементов на вход
model.add(Dense(400, activation='linear'))
model.add(Dense(10, activation='softmax')) #Функция активации softmax преобразует каждый элемент в вещественное число, при этом в сумме все они будут давать 1

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #Компилируем модель, metrics - используется для вывода оценочной информации, оптимайзер (как изменять веса) optimizer ='adam' с шагом обучения = 0.001 (для алгоритма обратного распространения ошибки), loss - функция ошибок (чтобы оценить, насколько различаются полученные данные и эталонные)
model.load_weights('model (1).h5') 

bot = telebot.TeleBot('1376184528:AAE8LTjrMpM4JyA-dDwf03F4WUH4zt2qNG0')
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, ты написал мне /start')

@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text == 'Привет':
        bot.send_message(message.chat.id, 'Привет, мой создатель')
    elif message.text == 'Пока':
        bot.send_message(message.chat.id, 'Прощай, создатель')

@bot.message_handler(content_types=['photo'])
def send_text(message):
  fileID = message.photo[-1].file_id
  file_info = bot.get_file(fileID)
  imgQ = bot.download_file(file_info.file_path)
  src = '/content/' + 'tmp.jpg'
  with open(src, 'wb') as new_file:
    new_file.write(imgQ)
  img = image.load_img(src, target_size=(28, 28), color_mode = 'grayscale')
  img = ImageOps.autocontrast(img)
  img = ImageOps.invert(img)
  num = image.img_to_array(img)
  num = num.reshape(1, 784)
  num = num.astype('float32')
  num = num / 255
  prediction = model.predict(num) #Получаем выходные значения модели
  pred = np.argmax(prediction) #Получаем индекс самого большого элемента (это итогования цифра, которую распознала сеть)'''
  bot.send_message(message.chat.id, pred)

bot.polling()