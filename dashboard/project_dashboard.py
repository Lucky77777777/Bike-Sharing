# Library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
from sklearn.linear_model import LinearRegression

# Data
df_day0 = pd.read_csv("https://raw.githubusercontent.com/Lucky77777777/Bike-Sharing/main/Dataset/df_day%20(clean).csv")
df_hour0 = pd.read_csv("https://raw.githubusercontent.com/Lucky77777777/Bike-Sharing/main/Dataset/df_hour%20(clean).csv")

# Ubah Format Tanggal
df_day0['dteday'] = pd.to_datetime(df_day0["dteday"])
df_hour0['dteday'] = pd.to_datetime(df_hour0["dteday"])

### Semua Fungsi
# Data berdasarkan periode waktu
def f_date(df, start_date, end_date):
  return df[(df["dteday"] >= str(start_date)) & (df["dteday"] <= str(end_date))]

# Lineplot jumlah peminjaman
def plot_bike_rental(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(25, 12))

    # Mengurutkan dataframe berdasarkan kolom 'dteday'
    df_sorted = df.sort_values('dteday')

    ax.plot(df_sorted['dteday'], df_sorted['casual'], linestyle='--', color='skyblue', label='Casual')
    ax.plot(df_sorted['dteday'], df_sorted['registered'], linestyle='--', color='salmon', label='Registered')
    ax.plot(df_sorted['dteday'], df_sorted['cnt'], linestyle='-', color='forestgreen', label='Total')

    ax.legend(['Casual', 'Registered', 'Total'], fontsize='large')

    return fig

# Rata-rata pinjaman berdasarkan musim
def f_season_cnt(df):
  return df.groupby(by="season").agg({
      "cnt": "mean"
  }).sort_values(by="cnt", ascending = False).reset_index()

# Rata-rata pinjaman berdasarkan waktu
def f_time_cnt(df):
  return df.groupby(by="time_of_day").agg({
      "cnt": "mean"
  }).sort_values(by="cnt", ascending = False).reset_index()

# Heatmap
def plot_heatmap_corr(df):
    plt.style.use('dark_background')

    df = df.rename(columns={'cnt': 'total_rental', 'hum': "kelembapan", "windspeed": "kecepatan_angin", "temp": "suhu"})

    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(corr_matrix, ax=ax, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)

    return fig

# Barplot Musim
def plot_season_cnt(season_cnt):
    plt.style.use('dark_background')

    season_cnt_sorted = season_cnt.sort_values(by='cnt', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(season_cnt_sorted['season'], season_cnt_sorted['cnt'], color='skyblue')

    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                round(bar.get_width(), 2),
                va='center', ha='right', fontsize=10, color='black')

    ax.set_xlabel('Jumlah Peminjaman')

    return fig

# Barplot Waktu
def plot_time_cnt(time_cnt):
    plt.style.use('dark_background')

    time_cnt_sorted = time_cnt.sort_values(by='cnt', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(time_cnt_sorted['time_of_day'], time_cnt_sorted['cnt'], color='skyblue')

    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                round(bar.get_width(), 2),
                va='center', ha='left', fontsize=10, color='black')

    ax.set_xlabel('Jumlah Peminjaman')
    ax.invert_xaxis()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    return fig

# Regresi Linear
def lr_plot(df, varx='temp', xlab='Suhu'):
    x = df[varx].values.reshape((-1, 1))
    y = df['cnt'].values

    # Model
    model = LinearRegression()
    model.fit(x, y)

    r2 = model.score(x, y)
    coefficients = model.coef_
    intercept = model.intercept_

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[varx], df['cnt'])

    num = len(df)
    start = df[varx].min()
    end = df[varx].max()
    xseq = np.linspace(start, end, num=num)

    ax.plot(xseq, intercept + coefficients[0] * xseq, color="orange", lw=1.5) 

    beta0 = r'$\mathbf{intercept = \hat\beta_0 =}$' + str(round(intercept, 2))
    beta1 = r'$\mathbf{slope = \hat\beta_1 =}$' + str(round(coefficients[0], 2))
    r_squared = r'$\mathbf{R^2 =}$' + str(round(r2, 2))

    ax.text(start, df['cnt'].max(), beta0, fontsize=10, color='orange', fontweight='bold', verticalalignment='top')
    ax.text(start, df['cnt'].max() - (df['cnt'].max() - df['cnt'].min()) / 10, beta1, fontsize=10, color='orange', fontweight='bold', verticalalignment='top')
    ax.text(start, df['cnt'].max() - (df['cnt'].max() - df['cnt'].min()) / 5, r_squared, fontsize=10, color='orange', fontweight='bold', verticalalignment='top')

    ax.set_xlabel(xlab)
    ax.set_ylabel('Jumlah Peminjaman')

    return fig

# Sidebar
with st.sidebar:
    # st.title("Bike Sharing")
    st.image("https://github.com/Lucky77777777/Bike-Sharing/raw/main/bike_icon.png", width=250)
    st.caption("Copyright vectips.com")

    min_date = df_day0["dteday"].min()
    max_date = df_day0["dteday"].max()

    start_date, end_date = st.date_input(
      label = 'Periode Peminjaman Sepeda', min_value = min_date,
      max_value = max_date,
      value = [min_date, max_date]
    )

df_day = f_date(df_day0, start_date, end_date)
df_hour = f_date(df_hour0, start_date, end_date)

### Main Board
st.markdown("<h1 style='color: skyblue;'> BIKE SHARING DASHBOARD📊 </h1>", unsafe_allow_html=True)

# Ringkasan
col1, col2, col3 = st.columns(3)

with col1:
   st.metric("Casual Users", value = df_day.casual.sum())
   st.caption("Avg. " + str(round(df_day.casual.mean())) + ' users/day')

with col2:
   st.metric("Registered Users", value = df_day.registered.sum())
   st.caption("Avg. " + str(round(df_day.registered.mean())) + ' users/day')

with col3:
   st.metric("Total Users", value = df_day.cnt.sum())
   st.caption("Avg. " + str(round(df_day.cnt.mean())) + ' users/day')

# Tren Peminjaman
st.header("Tren Jumlah Peminjaman Sepeda")
st.pyplot(plot_bike_rental(df_day))

season_dict = {1: "🌸 Springer", 2: "☀️ Summer", 3: "🍂 Fall" , 4: "❄️ Winter"}
cuaca_dict = {
    1: "🌤️ Clear, Few clouds, Partly cloudy, Partly cloudy",
    2: "🌫️ Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
    3: "❄️⛈️ Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
    4: "🌧️🌨️ Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"
}

tab1, tab2 = st.tabs(["Tertinggi", "Terendah"])

with tab1:
   st.subheader("Jumlah Peminjaman Tertinggi")
   max0 = df_day.loc[df_day['cnt'].idxmax()]

   st.text("📅 " + str(max0['dteday'].strftime('%Y-%m-%d')))
   st.text(season_dict.get(max0['season'], "Unknown") + ' Season')
   st.text(cuaca_dict.get(max0['weathersit'], "Unknown"))

   col1, col2, col3 = st.columns(3)

   with col1:
     st.metric("Casual Users", value = max0.casual)
   with col2:
     st.metric("Registered Users", value = max0.registered)

   with col3:
     st.metric("Total Users", value = max0.cnt)

   col1, col2, col3 = st.columns(3)

   with col1:
     st.metric("Suhu (°C)", value = round(max0['temp'], 2))

   with col2:
     st.metric("Kelembapan (%)", value = round(max0['hum'], 2))

   with col3:
     st.metric("Kecepatan Angin (km/h)", value = round(max0['windspeed'], 2))

with tab2:
   st.subheader("Jumlah Peminjaman Terendah")
   min0 = df_day.loc[df_day['cnt'].idxmin()]

   st.text("📅 " + str(min0['dteday'].strftime('%Y-%m-%d')))
   st.text(season_dict.get(min0['season'], "Unknown") + ' Season')
   st.text(cuaca_dict.get(min0['weathersit'], "Unknown"))

   col1, col2, col3 = st.columns(3)

   with col1:
     st.metric("Casual Users", value = min0.casual)
   with col2:
     st.metric("Registered Users", value = min0.registered)

   with col3:
     st.metric("Total Users", value = min0.cnt)

   col1, col2, col3 = st.columns(3)

   with col1:
     st.metric("Suhu (°C)", value = round(min0['temp'], 2))

   with col2:
     st.metric("Kelembapan (%)", value = round(min0['hum'], 2))

   with col3:
     st.metric("Kecepatan Angin (km/h)", value = round(min0['windspeed'], 2))

## Rata-Rata Peminjaman
st.header("Rata-Rata Jumlah Peminjaman Harian")

# Berdasarkan Musim
season_cnt = f_season_cnt(df_day)
season_cnt['cnt'] = round(season_cnt['cnt'])
season_cnt['season'] = season_cnt.season.replace({1: "Springer", 2: "Summer", 3: "Fall", 4: "Winter"})

# Berdasarkan Waktu
time_cnt = f_time_cnt(df_hour)
time_cnt['cnt'] = round(time_cnt['cnt'])

tab1, tab2 = st.tabs(["Musim", "Waktu"])

with tab1:
   st.subheader("Musim")
   st.pyplot(plot_season_cnt(season_cnt))

with tab2:
   st.subheader("Waktu")
   st.pyplot(plot_time_cnt(time_cnt))

## Pengaruh Cuaca
st.header('Pengaruh Cuaca Terhadap Jumlah Pinjaman')

# Heatmap
st.subheader('Heatmap Correlation')
fig = plot_heatmap_corr(df_day[["temp", "hum", "windspeed", "cnt"]])
st.pyplot(fig)

with st.expander("Penjelasan"):
  st.write("**Heatmap correlation** adalah representasi visual dari matriks korelasi antara variabel-variabel dalam sebuah dataset, di mana warna yang lebih gelap menunjukkan korelasi yang lebih kuat. Nilai korelasi berkisar dari -1 hingga 1, dengan nilai positif menandakan hubungan linier positif dan nilai negatif menandakan hubungan linier negatif antara variabel-variabel tersebut. Tanda + dan - pada korelasi menunjukkan arah hubungan antara variabel: positif menandakan hubungan yang searah, sementara negatif menandakan hubungan yang berlawanan arah.")

# Regresi Linear
st.subheader('Simple Linear Regression')
tab1, tab2, tab3 = st.tabs(["Suhu", "Kelembapan", "Kecepatan Angin"])

with tab1:
   st.pyplot(lr_plot(df_day, varx = 'temp', xlab = 'Suhu'))

with tab2:
   st.pyplot(lr_plot(df_day, varx = 'hum', xlab = "Kelembapan"))

with tab3:
   st.pyplot(lr_plot(df_day, varx = 'windspeed', xlab = "Kecepatan Angin"))

with st.expander("Penjelasan"):
  st.write("""
  **Simple Linear Regression** adalah metode statistik yang digunakan untuk memodelkan hubungan linier antara satu variabel independen (X) dan satu variabel dependen (Y). Dalam model ini, terdapat dua parameter utama: β₀ (intercept) dan β₁ (slope). Intercept (β₀) adalah titik di mana garis regresi memotong sumbu Y ketika nilai X adalah 0, sedangkan slope (β₁) menggambarkan kecenderungan perubahan Y ketika nilai X bertambah satu satuan. 
  
  Lebih lanjut, untuk mengevaluasi seberapa baik model memfitting data, digunakan nilai R² (R-squared). Nilai R² mengukur seberapa dekat data yang diobservasi dengan garis regresi yang telah dihasilkan oleh model. Nilai R² berkisar antara 0 hingga 1, di mana semakin tinggi nilai R², semakin baik model mampu menjelaskan variasi dalam data. Jika R² mendekati 1, itu menandakan bahwa model tersebut sangat cocok dengan data. Namun, jika nilai R² mendekati 0, model mungkin tidak cocok dengan data.
  """)
