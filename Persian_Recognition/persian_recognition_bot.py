import cv2
import telebot
import numpy as np
from keras.models import load_model
from retinaface import RetinaFace
import matplotlib.pyplot as plt


model = load_model("model.h5")
bot = telebot.TeleBot("___________YOUR_TOKEN____________")

width = 224
height = 224

@bot.message_handler(commands=['start'])
def say_hello(messages):
    bot.send_message(messages.chat.id, f'Wellcome Dear {messages.from_user.first_name}🌹')
    bot.send_message(messages.chat.id, f'Here you can distinguish Iranian from non-Iranian')
    bot.send_message(messages.chat.id, f'Now send me the photo so I can tell you😉')

@bot.message_handler(content_types=['photo'])
def photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = file_info.file_path
    with open("/content/" + src, 'wb') as new_file:
        new_file.write(downloaded_file)

    face = RetinaFace.extract_faces(img_path = "/content/" + src, align = True)
    face = cv2.cvtColor(face[0], cv2.COLOR_BGR2RGB)
    cv2.imwrite("/content/" + src, face)
    image = cv2.imread("/content/" + src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = image/255
    image = image.reshape(1, width, height, 3)
    pred = model.predict([image])

    res = np.argmax(pred)
    if res == 0:
      bot.reply_to(message, 'Iranian🇮🇷')
    else:
      bot.reply_to(message, 'Foreign')

bot.polling()