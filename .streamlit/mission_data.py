import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# UI Streamlit

st.set_page_config(
    page_title="Mon application Streamlit",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Image Background (marche pas ...)

page_bg_img = '''
<style>
body {
background-image: url("https://imgur.com/prCO34n");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Chargement des données + cache
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\\Users\\Ulysse\\Downloads\\data_bonheur_main.csv')
    return df

# Graphe Bonheur et Satisfaction (carte intéractive)

html_code = """
<iframe src="https://ourworldindata.org/grapher/happiness-cantril-ladder?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>
"""
st.write(html_code, unsafe_allow_html=True)

# Fonction pour obtenir les moyennes des caractéristiques 

def get_region_averages(df, happiness_score):
    happy_df = df[df['happiness_score'] == happiness_score]

    if not happy_df.empty:
        region = happy_df['region'].iloc[0]
        region_df = df[df['region'] == region]

        columns_to_average = ['economy_gdp_per_capita', 'family', 'health_life_expectancy']
        averages = region_df[columns_to_average].mean()
        return averages
    else:
        return None


df = load_data()

st.title("Bienvenue sur notre Streamlit !")
st.write("Ceci est une application Streamlit simple pour explorer les données sur le bonheur.")

selected_tab = st.radio(
    "Navigateur",
    ["Accueil", "Quel serait votre happiness_score ?", "Dans quel pays seriez-vous d'après votre happiness_score ?"]
)

# Partie prédiction de région selon le happiness_score

if selected_tab == "Dans quel pays seriez-vous d'après votre happiness_score ?":
    
    scaler_minmax = MinMaxScaler()

    X = df[['happiness_score', 'economy_gdp_per_capita', 'family', 'health_life_expectancy']]
    y = df['region']

    X_normalized = scaler_minmax.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, train_size=0.70, random_state=42)

    # Init + Training KNN

    modelKNN = KNeighborsClassifier()
    modelKNN.fit(X_train, y_train)

    @st.cache_data
    def prediction_region(happiness_score, economy_gdp_per_capita, family, health_life_expectancy):
        
        input_data = [[happiness_score, economy_gdp_per_capita, family, health_life_expectancy]]
        normalized_input = scaler_minmax.transform(input_data)
        
        predicted_region = modelKNN.predict(normalized_input)[0]
        return predicted_region

    choix_happiness_score = st.selectbox('Choisissez un happiness_score:', sorted(df['happiness_score'].unique()))

    if choix_happiness_score:
        moyennes = get_region_averages(df, choix_happiness_score)

        if moyennes is None:
            st.write("Aucune région trouvée pour ce happiness_score.")
        else:

            predicted_region = prediction_region(choix_happiness_score, moyennes['economy_gdp_per_capita'], moyennes['family'], moyennes['health_life_expectancy'])
            st.subheader(f"Pour un happiness_score de {choix_happiness_score:.2f}, la région prédite est {predicted_region}")

# Partie RL pour le happiness_score

if selected_tab == "Quel serait votre happiness_score ?":
   
    choix = st.text_input('Choisissez un pays:')

    if choix:
        moyennes = get_region_averages(df, choix)

        if moyennes is None:
            st.write("Aucune région trouvée pour ce happiness_score.")
        else:
            X = df[['economy_gdp_per_capita', 'family', 'health_life_expectancy']]
            y = df['happiness_score']

            # Ajouter 'happiness_score' (normalisation)
            X['happiness_score'] = y

            # Initialisation MMS
            
            scaler_minmax = MinMaxScaler()

            # Transformation des données

            normalized = scaler_minmax.fit_transform(X)

            X_normalized = pd.DataFrame(normalized, columns=X.columns)

            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, train_size=0.70, random_state=42)

            # Init + training ML

            model = LinearRegression()
            model.fit(X_train, y_train)

            df['predict'] = model.predict(X_normalized)

            predi = model.predict([moyennes.values[:-1]])[0]
            st.subheader(f"Vous habitez {choix}: votre happiness_score pourrait s'élever à {predi:.2f}")

# Affichage heatmap 

st.subheader('Heatmap des corrélations')
numeric_data = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 8))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, cmap="coolwarm", linewidths=0.5)
plt.title('Heatmap des corrélations')

st.pyplot(plt)

# Sidebar pour prédiction "rapide"

st.sidebar.header("Prédiction de région")
happiness_score = st.sidebar.slider("Happiness Score", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
economy_gdp_per_capita = st.sidebar.slider("Economy GDP per Capita", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
family = st.sidebar.slider("Family", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
health_life_expectancy = st.sidebar.slider("Health Life Expectancy", min_value=0.0, max_value=2.0, step=0.01, value=1.0)

if st.sidebar.button("Faire la prédiction"):
    pred_region = prediction_region(happiness_score, economy_gdp_per_capita, family, health_life_expectancy)
    st.sidebar.write(f"Prédiction de région : {pred_region}")

# Classement années + affichage des graphes

df_sorted = df.sort_values(by="year")

# Graphique 1 : Happiness Score par PBI (plotlyex)

fig1 = px.scatter(df_sorted,
                  x="economy_gdp_per_capita",
                  y="happiness_score",
                  title="HAPPINESS_SCORE PAR PBI",
                  color="region",
                  animation_frame='year',
                  size="happiness_score",
                  color_continuous_scale='viridis')
st.plotly_chart(fig1)

# Graphique 2 : Happiness Score/Family (plotly ex)

fig2 = px.scatter(df_sorted,
                  x="health_life_expectancy",
                  y="happiness_score",
                  title="HAPPINESS_SCORE PAR ESPERANCE DE VIE",
                  color="region",
                  animation_frame='year',
                  size="happiness_score",
                  color_continuous_scale='viridis')
st.plotly_chart(fig2)

# Graphique 3 : Happiness Score/Soutien social (plotly ex)

fig3 = px.scatter(df_sorted,
                  x="family",
                  y="happiness_score",
                  title="HAPPINESS_SCORE PAR SOUTIEN SOCIAL",
                  color="region",
                  animation_frame='year',
                  size="happiness_score",
                  color_continuous_scale='viridis')
st.plotly_chart(fig3)
