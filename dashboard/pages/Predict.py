import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import load

knn = load("C:/Users/MSI/Desktop/OmSTU/MachineLearning/dashboard/dash_models/knn.joblib")
bagging = load("C:/Users/MSI/Desktop/OmSTU/MachineLearning/dashboard/dash_models/bagging.joblib")
neyron = load("C:/Users/MSI/Desktop/OmSTU/MachineLearning/dashboard/dash_models/neyron.joblib")

models = {"k ближайших соседей" : knn, "Бэггинг" : bagging, "Нейронная сеть" : neyron}

select_model = st.selectbox(
    'Модель:',
    ("k ближайших соседей", "Бэггинг", "Нейронная сеть"))

model = models[select_model]

st.title('Ввод данных')
distance_from_home = st.slider('Дистанция, на которой произошла транзакция, от дома:',
                               1., 500., 1.)
distance_from_last_transaction = st.slider('Дистанция от места последней транзакции:',
                               1., 250., 1.)
ratio_to_median_purchase_price = st.slider('Отношение цены покупки к средней цене всех покупок:',
                               0., 60., 0.)
repeat_retailer = st.selectbox(
    'Повторяющиеся транзакции от одного продавца:',
    (True, False))
used_chip = st.selectbox(
    'Транзакция с кредитной карты или чипа:',
    (True, False))
used_pin_number = st.selectbox(
    'В момент транзакции использован PIN-код:',
    (True, False))
online_order = st.selectbox(
    'Онлайн-заказ:',
    (True, False))

st.title("Заключение")
X = np.array([[distance_from_home,
              distance_from_last_transaction,
              ratio_to_median_purchase_price,
              repeat_retailer,
              used_chip,
              used_pin_number,
              online_order]])
pred = int(model.predict(X)[0])

X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

if pred != 0:
    conclusion = "Вероятнее всего, транзакция является мошеннической."
else:
    conclusion = "Вероятнее всего, транзакция является честной."

st.write(conclusion + " (" + str(pred) + ")")