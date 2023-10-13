# 기본 라이브러리
import os
import gc
import time
import random
import datetime
import subprocess
import logging
import shutil
import pickle

# 이미지 처리 관련 라이브러리
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 데이터 처리 및 계산 관련 라이브러리
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

# TensorFlow 및 Keras 관련 라이브러리
import tensorflow as tf
from tensorflow.keras import layers, preprocessing, utils, callbacks
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda, add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

# PyTorch 관련 라이브러리
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 설정 파일 관련 라이브러리 // 이 Ver 은 config를 사용하지 않지만 사용할때를 대비
from configparser import ConfigParser

# TRAIN or PICK or TUNE 으로 OPERATION을 입력한다.
OPERATION = 'PICK'

# 모델을 학습할때 이 사이즈와 동일한 사이즈로 전처리 후 학습해야 하며, 여기 입력은 새로운 이미지를 비교할때 메모리상 Resize를 하는 용도임.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 960

# 데이터 및 모델 설정 ( 이곳에 직접 입력, python 이 독립적으로 작동 될 수 있도록 한다 )
BASE_DIR = 'C:\\Users\\enmus\\FindHolyGrail'
MODEL_FILE_PATH = 'HolyGrail_1006_01_Euclidean.keras'

# TRAIN Setting
CREATE_NEW_MODEL = True #True or False
EPOCHS = 8 # 통상 5 ~ 8 Epochs 내외에서 학습률 하락됨 
batch_size = 32 # 배치사이즈 , 12GB VRAM + 64G DDR4 에서 16,32 가능 2 의 제곱수로 셋팅을 권고합니다.

# PICK Setting
START_PICK = 7777
END_PICK = 7777

# TUNE Setting

### 1005 Vector Embedding , 샴 네트워크 방식 구현 ###
### 1006 완성도를 높이기 위한 정리 작업중 ( Freeze 별도 존재 )

ImageFile.LOAD_TRUNCATED_IMAGES = True
# GPU 제대로 사용하는지 확인 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 학습데이타 준비 Class
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, positive_data_folder, negative_data_folder, batch_size=32, is_validation=False):
        self.positive_data_folder = positive_data_folder
        self.negative_data_folder = negative_data_folder
        self.batch_size = batch_size
        self.positive_data = self.load_data(self.positive_data_folder)
        self.negative_data = self.load_data(self.negative_data_folder)
    
        split_idx = int(len(self.positive_data) * 0.8)
        if is_validation:
            self.positive_data = self.positive_data[split_idx:]
            self.negative_data = self.negative_data[split_idx:]

    def load_data(self, folder_path):
        img_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.png') or fname.endswith('.jpg')]
        return img_paths

    def __len__(self):
        # return int(np.ceil(len(self.positive_data) / self.batch_size))
        return len(self.positive_data)

    def __getitem__(self, index):
        X1_batch = []  
        X2_batch = []  
        y_batch = []
        pos_img = np.array(Image.open(self.positive_data[index]))
        # 긍정-부정 셋 생성
        for _ in range(self.batch_size // 2):
            neg_index = np.random.choice(len(self.negative_data))
            X1_batch.append(pos_img)
            X2_batch.append(np.array(Image.open(self.negative_data[neg_index])))
            y_batch.append(0)
        
        # 긍정-긍정 셋 생성
        for _ in range(self.batch_size // 2):
            while True:
                pos_index2 = np.random.choice(len(self.positive_data))
                if pos_index2 != index:
                    break
            X1_batch.append(pos_img)
            X2_batch.append(np.array(Image.open(self.positive_data[pos_index2])))
            y_batch.append(1)

        return [np.array(X1_batch, dtype='float32'), np.array(X2_batch, dtype='float32')], np.array(y_batch, dtype='int8')

# LOSS 함수 (Embedding Vector용)
def contrastive_loss(y_true, y_pred, margin=1.0):
    # Hadsell-et-al.'06에서 제안한 Contrastive loss 두 임베딩간의 거리를 기반으로 하는 손실 함수
    
    y_true = tf.cast(y_true, tf.float32)
    # 두 임베딩의 거리 계산
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    
    # Contrastive 손실 계산
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Resize_image, 학습된 모델의 사이즈와 동일해야함. 
def resize_image(image, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT):
    # Aspect ratio와 목표 차원을 기반으로 새로운 차원을 계산
    aspect_ratio = image.width / image.height
    if image.width > image.height:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    # 새 이미지 객체 생성 및 중앙에 원본 이미지 붙여넣기
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    new_image.paste(image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return np.array(new_image, dtype='float32')

def get_embedding_filename(model_filepath):
    base_name = os.path.basename(model_filepath)
    name_without_extension = os.path.splitext(base_name)[0]
    return name_without_extension + '.pkl'

# 임베딩 시각화 함수
def visualize_embeddings(embeddings, labels):
    perplexity_value = min(30, embeddings.shape[0] - 1)  
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], c='b', label='Positive')
    plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], c='r', label='Negative')
    plt.legend()
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE visualization of embeddings')
    plt.show()
    plt.savefig("embedding_visualization.png", format='png', dpi=300)

# 임베딩을 활용한 벡터 추출
def compute_embeddings(model, positive_images, filename):
    embeddings = []
    num_images = len(positive_images)
    for idx, img_path in enumerate(positive_images):
        image = np.array([resize_image(Image.open(img_path))], dtype='float32')
        embedding_outputs = model.predict([image, image])
        embedding = embedding_outputs[0][0]
        embeddings.append(embedding)
        print(f"\rProcessing image {idx + 1} out of {num_images}", end="")
    print("\n Embedding Vector 값이 계산되어 저장되었습니다.")
    
    # Pickle을 사용하여 임베딩 벡터 리스트 저장
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

# 유사도 평가 변경 ( 파일로 저장하지 않고 PICK할때 인스턴스하게 확인 )
def evaluate_similarity(new_embedding, embeddings, num_samples=12, exclude_top=2, num_avg=5):
    def compute_random_avg_embedding(embeddings):
        indices = np.arange(len(embeddings))  # 임베딩 리스트의 인덱스 배열 생성
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        selected_embeddings = [embeddings[i] for i in selected_indices]

        distances = [np.linalg.norm(e - np.mean(selected_embeddings, axis=0)) for e in selected_embeddings]
        sorted_indices = np.argsort(distances)
        top_embeddings = [selected_embeddings[i] for i in sorted_indices[:-exclude_top]]
        avg_embedding = np.mean(top_embeddings, axis=0)
        
        return avg_embedding
    avg_embeddings = [compute_random_avg_embedding(embeddings) for _ in range(num_avg)]
    distances = [np.linalg.norm(new_embedding - avg_embedding) for avg_embedding in avg_embeddings]
    return min(distances)

def get_embedding_model(model, input_shape):
    # 주어진 샴 네트워크에서 이미 정의된 기본 네트워크(base_network)를 가져옵니다.
    base_network = model.layers[2]
    # 새로운 입력 레이어 생성
    input_layer = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    # 임베딩 계산및 모델 반환
    embeddings = base_network(input_layer)
    return tf.keras.models.Model(inputs=input_layer, outputs=embeddings)

#ResNet-50 모델을 사용하여 추가 레이어
def Train_HolyGrail(input_shape):
    # 사전 학습된 ResNet-50 모델 불러오기
    def create_base_model():
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # ResNet-50 모델의 레이어를 동결 (추가 학습을 위해 필요한 경우 변경 가능)
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        embedding_output = tf.keras.layers.Dense(128)(x)  # 활성화 함수 없음
        return tf.keras.models.Model(inputs=base_model.input, outputs=embedding_output)
    
    # 두 이미지를 입력으로 받을 수 있도록 입력 레이어를 정의합니다.
    input_image1 = tf.keras.layers.Input(input_shape)
    input_image2 = tf.keras.layers.Input(input_shape)
    
    base_network = create_base_model()
    
    embedding_1 = base_network(input_image1)
    embedding_2 = base_network(input_image2)
    
    siamese_network = tf.keras.models.Model(inputs=[input_image1, input_image2], outputs=[embedding_1, embedding_2])
    return siamese_network

def process_pick(target_folder, model):  
    candidate_folder = os.path.join(target_folder, "candidate")
    pick_by_ai_folder = os.path.join(target_folder, "PickByAi")
    os.makedirs(pick_by_ai_folder, exist_ok=True)

    # 지정된 파일 이름에서 임베딩 벡터 불러오기
    embedding_filename = get_embedding_filename(MODEL_FILE_PATH)
    with open(embedding_filename, 'rb') as f:
        embeddings = pickle.load(f)

    candidate_images = [os.path.join(candidate_folder, f) for f in os.listdir(candidate_folder) if os.path.isfile(os.path.join(candidate_folder, f))]
    total_files = len(candidate_images)
    picked_images = []
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    embedding_model = get_embedding_model(model, input_shape)

    for idx, img_path in enumerate(candidate_images):
        print(f"Processing file {idx + 1} out of {total_files}...", end='\r')
        image = resize_image(Image.open(img_path))
        image_np = np.array([image], dtype='float32')
        new_embedding = embedding_model.predict(image_np)[0]
        distance = evaluate_similarity(new_embedding, embeddings)
        # 거리를 파일 이름에 포함시켜 저장
        new_file_name = f"{distance:.4f}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(pick_by_ai_folder, new_file_name))
        
    return len(candidate_images)  # 선택된 이미지의 수 반환

# TUNE 모드 함수 ( 오버 피팅 가능성이 높아서 Embedding 방식에서 Fine-Tune은 조금더 검증해야 함)
def finetune(target_folder, model, batch_size_finetune=16, learning_rate=0.0001):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=contrastive_loss)
    
    # 이미지 불러오기
    success_images = [os.path.join(target_folder, 'success', f) for f in os.listdir(os.path.join(target_folder, 'success'))]
    fail_images = [os.path.join(target_folder, 'fail', f) for f in os.listdir(os.path.join(target_folder, 'fail'))]

    for pos_img_path in success_images:
        pos_img = resize_image(Image.open(pos_img_path))

        # 긍정-긍정 셋 구성
        positive_samples = np.random.choice([img for img in success_images if img != pos_img_path], batch_size_finetune // 2, replace=False)
        X1_pos = np.array([pos_img] * (batch_size_finetune // 2))
        X2_pos = np.array([resize_image(Image.open(img_path)) for img_path in positive_samples])

        # 긍정-부정 셋 구성
        negative_samples = np.random.choice(fail_images, batch_size_finetune // 2, replace=False)
        X1_neg = np.array([pos_img] * (batch_size_finetune // 2))
        X2_neg = np.array([resize_image(Image.open(img_path)) for img_path in negative_samples])

        X1_batch = np.concatenate([X1_pos, X1_neg], axis=0)
        X2_batch = np.concatenate([X2_pos, X2_neg], axis=0)
        y_batch = np.concatenate([np.ones(batch_size_finetune // 2), np.zeros(batch_size_finetune // 2)], axis=0)

        model.train_on_batch([X1_batch, X2_batch], y_batch)

    return model

def main():
    start_time = datetime.datetime.now()
    print(f"작업 시작 시각: {start_time}")

    if OPERATION == 'TRAIN':
        print(f"TRAIN 모드를 진행 합니다.")
        input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if CREATE_NEW_MODEL:
            model = Train_HolyGrail(input_shape)
        else:
            model = tf.keras.models.load_model(MODEL_FILE_PATH)
        model.compile(optimizer='adam', loss=contrastive_loss)
        positive_data_folder = os.path.join(BASE_DIR, 'TRAINPACK', 'PO')
        negative_data_folder = os.path.join(BASE_DIR, 'TRAINPACK', 'NG')

        # 학습,검증데이타 분리하여 Gen
        train_gen = DataGenerator(positive_data_folder, negative_data_folder, batch_size, is_validation=False)
        val_gen = DataGenerator(positive_data_folder, negative_data_folder, batch_size, is_validation=True)

        # 모델 학습
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
        model.save(MODEL_FILE_PATH)
        print(f"모델이 학습, 저장되었습니다: {MODEL_FILE_PATH}")

        # 학습이 완료된 후, 학습한 긍정이미지들의 모든 임베딩 벡터 .pkl 파일에 저장
        positive_images = [os.path.join(positive_data_folder, f) for f in os.listdir(positive_data_folder) if f.endswith('.png') or f.endswith('.jpg')]
        embedding_filename = get_embedding_filename(MODEL_FILE_PATH)
        compute_embeddings(model, positive_images, embedding_filename)

        # 비주얼라이제이션 코드 추가 예정
        del train_gen, val_gen
        gc.collect()

    elif OPERATION == 'PICK':
        print(f"PICK 모드를 진행 합니다.")

        model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects={"contrastive_loss": contrastive_loss})

        # pickle로 저장된 임베딩 벡터 로드
        with open('avg_embedding.pkl', 'rb') as f:
            positive_embedding_avg = np.array(pickle.load(f))
            print(positive_embedding_avg.shape)
            print(positive_embedding_avg)
            print(type(positive_embedding_avg))

        print(model.summary())  # 모델 구조 출력

        # Target 폴더 순회
        target_folders = [os.path.join(BASE_DIR, f"Target{str(i).zfill(4)}") for i in range(START_PICK, END_PICK + 1)]
        for target_folder in target_folders:
            num_images = process_pick(target_folder, model)
           
            # 메시지 출력 및 윈도우 탐색기 열기
            folder_name = os.path.basename(target_folder)
            print(f"{folder_name} 폴더에서 이미지를 총 {num_images} 개 찾았습니다.")
            subprocess.Popen(f'explorer {os.path.realpath(os.path.join(target_folder, "PickByAi"))}')

    elif OPERATION == 'TUNE' :
        print(f"FINETUNE 모드를 진행 합니다.")
        model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects={"contrastive_loss": contrastive_loss})
        # Target 폴더 순회
        target_folders = [os.path.join(BASE_DIR, f"Target{str(i).zfill(4)}") for i in range(START_PICK, END_PICK + 1)]
        for target_folder in target_folders:
            model = finetune(target_folder, model)

        model.save(MODEL_FILE_PATH)
        print(f"모델이 파인튜닝되었습니다: {MODEL_FILE_PATH}")

    end_time = datetime.datetime.now()
    print(f"작업 종료 시각: {end_time}")

if __name__ == "__main__":
    main()
