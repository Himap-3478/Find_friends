import streamlit as st
import pandas as pd
import json
from pycaret.clustering import load_model, predict_model
import plotly.express as px



MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v3.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r") as f:
        return json.loads(f.read())

        
@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

st.title("Znajdź znajomych")

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znależć osoby podobne do Ciebie")
    age = st.selectbox("Wiek",['<18', '19-25', '26-35', '36-45', '46-55', '56-65', '>65'])
    edu_level = st.selectbox("Wykształcenie",['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzę",['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce",['Nad wodą', 'W górach', 'W lesie', 'Inne'])
    gender = st.radio("Płeć",['Mężczyzna', 'Kobieta'])
    

    person_df = pd.DataFrame([
        {
            "age": age,
            "edu_level": edu_level,
            "fav_animals": fav_animals,
            "fav_place": fav_place,
            "gender": gender
        }
    ])


model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba Twoich znajomych", len(same_cluster_df))


fig = px.histogram(same_cluster_df, x="gender", color_discrete_sequence=['#1f77b4'])
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="age", color_discrete_sequence=['#ff7f0e'])
fig.update_layout(
    title="Wiek członków grupy",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals", color_discrete_sequence=['#2ca02c'])
fig.update_layout(
    title="Ulubione zwierzęta w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="fav_place", color_discrete_sequence=['#d62728'])
fig.update_layout(
    title="Ulubione miejsca w grupie",
    xaxis_title="Ulubione miejsca",
    yaxis_title="Liczba osób", 
)
st.plotly_chart(fig)

