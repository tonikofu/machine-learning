import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def heat_plot(predicators):
    fig, ax = plt.subplots()
    sns.heatmap(predicators.corr(),
                ax=ax)
    st.pyplot(fig)

def pair_plot(data):
    fig = sns.pairplot(data,
                       hue='fraud')
    st.pyplot(fig)

def dis_plot(data, feature):
    fig = sns.displot(data=data,
                      x=feature,
                      hue='fraud',
                      log_scale=True,
                      kde=True)
    st.pyplot(fig)

card_transdata = pd.read_csv('C:/Users/MSI/Desktop/OmSTU/MachineLearning/data/card_transdata.csv')
data_sampled = card_transdata.sample(10000)
target_feature = card_transdata["fraud"]
predicators = card_transdata.drop(["fraud"], axis=1)

select_plot = st.selectbox(
    'Select plot:',
    ("Pair Plot",
     "Heatmap",
     "Displot for distance_from_home",
     "Displot for distance_from_last_transaction")
    )

if select_plot == "Pair Plot":
    pair_plot(data_sampled)
elif select_plot == "Heatmap":
    heat_plot(predicators)
elif select_plot == "Displot for distance_from_home":
    dis_plot(card_transdata, "distance_from_home")
elif select_plot == "Displot for distance_from_last_transaction":
    dis_plot(card_transdata, "distance_from_last_transaction")
