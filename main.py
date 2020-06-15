import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from pdf2image import convert_from_path

from horizon import fix_horizon
from segmetation import lines_segmentation, word_segmentation, character_segmentation

MODEL_FILE = './models/model.h5'
CLASSES = ['(', ')', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '?', 'I', 'SLASH', 'Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
IMG_HEIGHT = 150
IMG_WIDTH = 150

def read_pdf_file(file_name='test.pdf'):
    pages = convert_from_path(file_name, 500)
    count = 0
    model = tf.keras.models.load_model(MODEL_FILE)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()

    output = ''
    
    for index, page in enumerate(pages):
        # Pre processing
        image = np.array(page)
        image = fix_horizon(image)
        # TODO: Add remove watermarks and printing
        # TODO: Fix blur
        # TODO: Filter by Text color?
        # TODO: remove long lines
        
        # Binarize
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binarized_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        for text_line in lines_segmentation(binarized_image[1]):
            count += 1
            # cv2.imshow("text_line", text_line)
            for word in word_segmentation(text_line):
                # cv2.imshow("word", word)
                previous_character = ''
                recognized_word = ''
                for character in character_segmentation(word):
                    # Generate Dataset
                    # cv2.imshow("character", character)
                    # cv2.waitKey(1000)
                    # cv2.waitKey(0)
                    # class_name = input('Print CLASS_NAME:')
                    # class_name = class_name.upper()
                    # cv2.imwrite(f"dataset/train/{class_name}/character_{count}.png", character)
                    # count += 1

                    # Reconize character with NN
                    result = cv2.resize(character, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
                    color_image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                    input_arr = np.array([color_image])
                    predictions = model.predict(input_arr)
                    print(predictions)
                    character_idx = tf.argmax(predictions[0]).numpy()
                    print(character_idx)
                    value = CLASSES[character_idx]
                    print(value)
                    if value == 'SLASH':
                        value = '/'
                    if value == 'DOT':
                        value = '.'
                    if previous_character == 'Ь' and value == 'I':
                        value = 'Ы'
                        recognized_word = recognized_word[:-1]
                    previous_character = value
                    recognized_word += value
                    print('recognized_word:', recognized_word)
                    # cv2.imshow("character", character)
                    # cv2.waitKey(0)
                output += f'{recognized_word} '
            output += '\n'
            print(f'LINE {count}')
        output += '\n'*10

    # TODO: add natural language processing
    with open("output.txt", "a") as f:
        f.write(output)
    
if __name__ == '__main__':
    read_pdf_file()
