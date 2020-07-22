import re
import collections
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

n = 0
max_len = 38
word_to_idx = None
idx_to_word = None
model = None


def readTextFile(path):
    with open(path) as f:
        captions = f.read()
    return captions


def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub('[^a-z]+', ' ', sentence)
    sentence = sentence.split()

    sentence = ' '.join(sentence)

    return sentence


class LSTM:

    def __init__(self):

        captions = readTextFile('models/Flickr8k.token.txt')
        captions = captions.split('\n')[:-1]

        descriptions = dict()

        for caption in captions:
            image_name = caption.split('\t')[0].split('.jpg#')[0]
            image_desc = caption.split('\t')[1]

            if descriptions.get(image_name) is None:
                descriptions[image_name] = []

            descriptions[image_name].append(image_desc)

        for key, descs in descriptions.items():
            for i in range(len(descs)):
                descs[i] = clean_text(descs[i])

        vocab = set()

        for key in descriptions.keys():
            [vocab.update(sentence.split()) for sentence in descriptions[key]]  # Unique Words

        total_words = []

        for key in descriptions.keys():
            [total_words.append(word) for senten in descriptions[key] for word in senten.split()]

        counter = collections.Counter(total_words)
        freq_cnts = dict(counter)

        print(len(freq_cnts))

        sorted_freq_cnts = sorted(freq_cnts.items(), reverse=True, key=lambda x: x[1])

        threshold = 10
        sorted_freq_cnts = [x for x in sorted_freq_cnts if x[1] > threshold]

        total_words = [x[0] for x in sorted_freq_cnts]

        self.word_to_idx = {}
        self.idx_to_word = {}

        for i, word in enumerate(total_words):
            self.word_to_idx[word] = i + 1
            self.idx_to_word[i + 1] = word  # Index 0 left to be used for padding for making all sentences equal

        self.idx_to_word[1851] = '<s>'
        self.word_to_idx['<s>'] = 1851

        self.idx_to_word[1852] = '<e>'
        self.word_to_idx['<e>'] = 1852

        self.model = load_model('models/model_39.h5')

        self.model_resnet = ResNet50(weights="imagenet", input_shape=(224, 224, 3))

        self.model_front = Model(self.model_resnet.input, self.model_resnet.layers[-2].output)

    def predict_caption(self, photo):
        photo = self.encode_img(photo)

        photo = photo.reshape((1, 2048))

        in_text = '<s>'

        for i in range(max_len):
            sequence = [self.word_to_idx[w] for w in in_text.split() if w in self.word_to_idx]

            sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

            y_pred = self.model.predict([photo, sequence])

            y_pred = y_pred.argmax()  # Greedy Sampling
            word = self.idx_to_word[y_pred]

            in_text += ' ' + word

            if word == '<e>':
                break
        final_captions = in_text.split()[1:-1]
        return ' '.join(final_captions)

    def preprocess_img(self, img):
        img = image.load_img(img, target_size=(224, 224))
        img = image.img_to_array(img)
        print(img.shape)
        img = tf.expand_dims(img, axis=0)
        print(img.shape)
        # Normalisation
        img = preprocess_input(img)
        return img

    def encode_img(self, img):
        img = self.preprocess_img(img)
        feature_vector = self.model_front.predict(img)

        feature_vector = feature_vector.reshape((-1,))

        return feature_vector
