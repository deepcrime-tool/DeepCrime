from keras.datasets import mnist
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import math

class Model():

    def get_test_data(self):
        pass

class MnistModel(Model):

    def input_reshape_test(self, x_test, y_test, num_classes):
        img_rows, img_cols = 28, 28
        num_classes = 10

        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_test = x_test.astype('float32')
        x_test /= 255

        return (x_test, y_test, keras.utils.to_categorical(y_test, num_classes))

    def get_test_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test, y_test, y_test_cl = self.input_reshape_test(self, x_train, y_train, 10)
        return x_test, y_test, y_test_cl

    def get_prediction_info(self, model_file):
        x_test, y_test, y_test_cl = self.get_test_data()
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_file)
                # scores = model.evaluate(x_test, y_test_cl)
                predictions = model.predict_classes(x_test)
        array = np.equal(predictions, y_test)
        return np.equal(predictions, y_test)


class MovieModel(Model):
    movielens_dir = '/home/ubuntu/mutation-tool/test_models/ml-latest-small/'

    def get_test_data(self):
        ratings_file = self.movielens_dir + "ratings.csv"
        df = pd.read_csv(ratings_file)
        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}
        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["movie"] = df["movieId"].map(movie2movie_encoded)

        num_users = len(user2user_encoded)
        num_movies = len(movie_encoded2movie)
        df["rating"] = df["rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(df["rating"])
        max_rating = max(df["rating"])

        print("Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(num_users, num_movies,
                                                                                                 min_rating,
                                                                                                 max_rating))
        df = df.sample(frac=1, random_state=42)
        x = df[["user", "movie"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
        # Assuming training on 90% of the data and validating on 10%.
        train_indices = int(0.9 * df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )
        print(x_train)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

        return x_train, y_train

    def get_model(self, model_file):
        EMBEDDING_SIZE = 50
        ratings_file = self.movielens_dir + "ratings.csv"
        df = pd.read_csv(ratings_file)

        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        userencoded2user = {i: x for i, x in enumerate(user_ids)}
        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["movie"] = df["movieId"].map(movie2movie_encoded)

        num_users = len(user2user_encoded)
        num_movies = len(movie_encoded2movie)

        modell = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
        modell.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
            metrics=['mse']
        )
        print(model_file)
        file = model_file.replace('.h5', '/') + 'movie_recomm_trained.h5py'
        print(os.path.exists(model_file.replace('.h5', '/')))
        modell.load_weights(file)
        return modell

    def get_prediction_info(self, model_file):
        x_test, y_test = self.get_test_data()
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = self.get_model(model_file)
                # scores = model.evaluate(x_test, y_test_cl)
                predictions = model.predict(x_test)

        array = np.isclose(predictions.flatten(), y_test, 0.12)
        return array

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)



class UnityModel(Model):
    def angle_loss_fn(self, y_true, y_pred):
        # print(y_true.shape)
        # print(y_pred.shape)
        x_p = math.sin(y_pred[:, 0]) * math.cos(y_pred[:, 1])
        y_p = math.sin(y_pred[:, 0]) * math.sin(y_pred[:, 1])
        z_p = math.cos(y_pred[:, 0])

        x_t = math.sin(y_true[:, 0]) * math.cos(y_true[:, 1])
        y_t = math.sin(y_true[:, 0]) * math.sin(y_true[:, 1])
        z_t = math.cos(y_true[:, 0])

        norm_p = math.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
        norm_t = math.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

        dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

        angle_value = dot_pt / (norm_p * norm_t)
        angle_value = tf.clip_by_value(angle_value, -0.99999, 0.99999)

        loss_val = (math.acos(angle_value))

        # tf.debugging.check_numerics(
        #     loss_val, "Vse propalo", name=None
        # )
        # print(loss_val.shape)
        return loss_val

    def get_test_data(self):
        dataset_folder = "\\mutation-tool\\datasets\\"
        x_img = np.load(dataset_folder + 'dataset_x_img.npy')
        x_head_angles = np.load(dataset_folder + 'dataset_x_head_angles_np.npy')
        y_gaze_angles = np.load(dataset_folder + 'dataset_y_gaze_angles_np.npy')

        x_img_train, x_img_test, x_ha_train, x_ha_test, y_gaze_train, y_gaze_test = train_test_split(x_img,
                                                                                                     x_head_angles,
                                                                                                     y_gaze_angles,
                                                                                                     test_size=0.2,
                                                                                                     random_state=42)
        return [x_img_test, x_ha_test], y_gaze_test

    def get_prediction_info(self, model_file):
        x_test, y_test = self.get_test_data()

        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_file, compile=False)
                # scores = model.evaluate(x_test, y_test_cl)
                predictions = model.predict(x_test)

        loss = self.angle_loss_fn(y_test, predictions)

        # print(loss.shape)
        check = (np.degrees(loss)<5)
        # print(check.shape)
        # print(check)
        return check


if __name__ == "__main__":
    U = UnityModel()
    model_path = ""
    check = U.get_prediction_info(model_path)