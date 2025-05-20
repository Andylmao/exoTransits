import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle
import base64
from matplotlib import rc

def area_interseccion_circulos(x1, y1, r1, x2, y2, z1, r2):
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d >= r1 + r2:
        return 0
    elif d <= abs(r1 - r2) and z1 < 0:
        return np.pi * (r2**2)
    elif d <= abs(r1 - r2) and z1 >= 0:
        return 0
    else:
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        return term1 + term2 - term3

def main():
    st.title("Simulación de Tránsito de Exoplaneta")

    df = pd.read_csv("https://raw.githubusercontent.com/Mastevegm/exoplanet.py/refs/heads/master/exoplaa.csv")

    st.sidebar.header("Parámetros de la Simulación")
    Radio_star = 10
    Rpf = st.sidebar.slider("Radio del planeta/radio de la estrella", 0.01, 0.4, 0.1, 0.01)
    Radio_planet = Rpf * Radio_star

    R_orbf = st.sidebar.slider("Radio orbital del planeta", 2.0, 10.0, 5.0, 0.1)
    Orbita = R_orbf * Radio_star

    Angulo_inclinacion = st.sidebar.slider("Ángulo de inclinación (grados)", -90, 90, 0, 1)
    Inclinacion = np.radians(90 + Angulo_inclinacion)

    Pasos = 800
    Caja = 1.5 * Orbita

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].set_xlim(-1 * Caja, Caja)
    axs[0].set_ylim(-1 * Caja, Caja)
    axs[0].set_aspect('equal')
    circle_central = plt.Circle((0, 0), Radio_star, color='yellow')
    axs[0].add_artist(circle_central)
    circle = plt.Circle((-1 * Orbita, 0), Radio_planet, color='red')
    axs[0].add_artist(circle)
    time_text = axs[0].text(0.05, 0.95, '', transform=axs[0].transAxes, color='black')

    axs[1].set_xlim(0, Pasos)
    axs[1].set_ylim(98, 101)
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Brillo (%)')
    area_dif_values = []
    line, = axs[1].plot([], [], 'r-')
    moving_circle = Circle((0, 100), 0.1, color='blue')
    axs[1].add_patch(moving_circle)

    cut_limits = [(2454956.9, 2454959.59)]
    for x_min, x_max in cut_limits:
        mask2 = (df['Tiempo'] >= x_min) & (df['Tiempo'] <= x_max)
        df_cut = df[mask2]
        brightness_scaled = df_cut['Flujo'] / (0.997 * df_cut['Flujo'].max()) * 100
        axs[1].scatter((df_cut['Tiempo'] - x_min) / (x_max - x_min) * Pasos, 
                      brightness_scaled, label=f'({x_min}, {x_max})', color='green')
    axs[1].legend()

    def init():
        circle.set_center((-1 * Orbita, 0))
        time_text.set_text('')
        line.set_data([], [])
        moving_circle.center = (0, 100)
        return circle, circle_central, time_text, line, moving_circle

    def animate(frame):
        x = np.cos(0.5*np.pi + 2 * np.pi * frame / Pasos) * Orbita
        y = np.sin(0.5*np.pi + 2 * np.pi * frame / Pasos) * Orbita * np.cos(Inclinacion)
        z = np.sin(0.5*np.pi + 2 * np.pi * frame / Pasos) * Orbita
        circle.set_center((x, y))
        if z > 0:
            circle.set_zorder(0)
            circle_central.set_zorder(1)
            area_diferencia = 100
        else:
            circle.set_zorder(1)
            circle_central.set_zorder(0)
            area_diferencia = 100 * (1 - (area_interseccion_circulos(0, 0, Radio_star, x, y, z, Radio_planet)) / (np.pi * Radio_star**2))
        time_text.set_text('Brillo: {:.2f}%'.format(area_diferencia))
        area_dif_values.append(area_diferencia)
        frames = list(range(len(area_dif_values)))
        line.set_data(frames, area_dif_values)
        if len(area_dif_values) > 0:
            moving_circle.center = (frame, area_dif_values[-1])
        else:
            moving_circle.center = (frame, 100)
        return circle, circle_central, time_text, line, moving_circle

    anim = animation.FuncAnimation(fig, animate, frames=Pasos, interval=1, 
                                   init_func=init, blit=False, repeat=True)

    # Mostrar la figura como imagen estática
    st.pyplot(fig)
    st.write("Animación del tránsito del exoplaneta")

    # Guardar animación como video y mostrarla
    rc('animation', html='html5')
    video_path = "/tmp/transit.mp4"
    anim.save(video_path, writer='ffmpeg', fps=30)

    with open(video_path, "rb") as f:
        video_bytes = f.read()
        b64_video = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <video controls autoplay loop>
        <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
