import cv2
import telebot
from keras.models import load_model

model = load_model("/content/model.h5")
bot = telebot.TeleBot("___________YOUR_TOKEN___________")

@bot.message_handler(commands=['start'])
def say_hello(messages):
    bot.send_message(messages.chat.id, f'Wellcome Dear {messages.from_user.first_name}🌹')
    bot.send_message(messages.chat.id, f'Here you can distinguish sheykh👳🏻‍♂️ from who are not sheikhs👨🏻👩🏻')
    bot.send_message(messages.chat.id, f'Now send me the photo so I can tell you😉')

@bot.message_handler(content_types=['photo'])
def photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = file_info.file_path
    with open("/content/" + src, 'wb') as new_file:
        new_file.write(downloaded_file)

    image = cv2.imread("/content/" + src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = image/255
    image = image.reshape(1, width, height, 3)
    pred = model.predict([image])

    res = np.argmax(pred)
    if res == 0:
      bot.reply_to(message, 'not sheykh 👨🏻👩🏻')
    else:
      bot.reply_to(message, 'sheykh 👳🏻‍♂️')

bot.polling()