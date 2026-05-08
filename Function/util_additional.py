import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def analisis_spektrum_source(source, dt, zoom_x=None):
    """
    Analisis spektrum frekuensi dari data source 1D.
    
    Parameters:
        source (np.ndarray): array 1D atau 2D (n_time, 1) dari source time signal
        dt (float): interval waktu (dalam detik)
    """
    # Jika source berdimensi (n_time, 1), ubah ke 1D
    if source.ndim == 2 and source.shape[1] == 1:
        source = source[:, 0]

    n = len(source)
    freq = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(source))

    # Frekuensi dominan
    f_dominan = freq[np.argmax(spectrum)]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(freq, spectrum, label='Spektrum')
    plt.axvline(f_dominan, color='r', linestyle='--', label=f'f_dominan = {f_dominan:.2f} Hz')
    if zoom_x is not None:
        plt.xlim(zoom_x[0], zoom_x[1])
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Amplitudo')
    plt.title('Spektrum Frekuensi Source')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return f_dominan

    
def dominant_frequency(data, dt, plot=True, zoom_x=None):
    """
    Menghitung frekuensi dominan dari data seismik (n_time, n_receiver).
    
    Parameters:
    -----------
    data : 2D np.array
        Data seismik gather (n_time, n_receiver).
    dt : float
        Sampling interval dalam detik.
    plot : bool
        Jika True, tampilkan spektrum frekuensi rata-rata.

    Returns:
    --------
    f_dom : float
        Frekuensi dominan dalam Hz.
    f : np.array
        Array frekuensi.
    spectrum : np.array
        Rata-rata spektrum amplitudo.
    """
    n_time, _ = data.shape
    fs = 1.0 / dt  # frekuensi sampling
    f = np.fft.rfftfreq(n_time, d=dt)

    # FFT tiap trace dan ambil magnitudo, lalu rata-ratakan
    spectrum = np.abs(np.fft.rfft(data, axis=0))
    avg_spectrum = np.mean(spectrum, axis=1)

    # Ambil frekuensi dominan (frekuensi dengan amplitudo maksimum)
    f_dom = f[np.argmax(avg_spectrum)]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(f, avg_spectrum, label='Rata-rata Spektrum')
        plt.axvline(f_dom, color='r', linestyle='--', label=f'Fdom = {f_dom:.2f} Hz')
        plt.xlabel('Frekuensi (Hz)')
        plt.ylabel('Amplitudo')
        if zoom_x is not None:
            plt.xlim(zoom_x[0], zoom_x[1])
        plt.title('Spektrum Frekuensi Rata-rata dari Gather')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return f_dom, f, avg_spectrum












def plot_shot_gather(receiver_coords, source_coords, shot_idx):
    # Ambil koordinat untuk satu shot
    recs = receiver_coords[shot_idx]  # shape: (nreceiver, 2)
    src = source_coords[shot_idx]     # shape: (2,)

    plt.figure(figsize=(10, 5))
    plt.scatter(recs[:, 0], recs[:, 1], c='blue', label='Receiver', marker='v')
    plt.scatter(src[0], src[1], c='red', label='Source', marker='*', s=100)
    plt.title(f'Shot {shot_idx}')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.gca().invert_yaxis()  # karena Z biasanya ke bawah dalam data seismik
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_all_shots(receiver_coords, source_coords):
    nshot = source_coords.shape[0]

    plt.figure(figsize=(12, 6))

    for shot in range(nshot):
        recs_x = receiver_coords[shot, :, 0]  # ambil hanya x receiver
        shot_num = [shot+1] * len(recs_x)       # y-nya: nomor shot yang sama

        # plot receiver
        plt.scatter(recs_x, shot_num, color='blue', s=5, label='Receiver' if shot == 0 else "")

        # plot source
        plt.scatter(source_coords[shot, 0], shot+1, color='red', marker='*', s=30, label='Source' if shot == 0 else "")

    plt.xlabel('X Coordinate')
    plt.ylabel('Shot Number')
    #plt.ylim(1,125)
    #plt.yticks()
    plt.title('Source & Receiver Coordinates per Shot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_wavefield_overlay(velocity, wavefield, dt, time_step, origin, spacing, shape, nboundary, shot_coordinate=False, receiver_coordinate=False, cmap_wave='seismic', alpha=0.3, show=False, save=False, path=None, SeisAmpMin=-1, SeisAmpMax=1):
    """
    Overlay wavefield snapshot pada model velocity.
    
    Parameters:
    - velocity: 2D array (n_x, n_z)
    - wavefield: 3D array (n_time, n_x, n_z)
    - time_step: waktu (index) yang ingin dipilih
    - cmap_wave: colormap untuk wavefield
    - alpha: transparansi overlay wavefield
    """
    origin = origin
    spacing = spacing
    shape = shape
    nboundary = nboundary
    dt=dt
    path=path

    fig, ax = plt.subplots(figsize=(10, 6))

    xmin = (origin[0]*spacing[0])/1000
    zmin = (origin[1]*spacing[1])/1000
    xmax = (origin[0]*spacing[0] + shape[0]*spacing[0])/1000
    zmax = (origin[1]*spacing[1] + shape[1]*spacing[1])/1000

    velocity = velocity.vp.data[nboundary:nboundary+shape[0], nboundary:nboundary+shape[1]]


    # Overlay wavefield pada waktu tertentu
    wave = wavefield.data[time_step][nboundary:nboundary+shape[0], nboundary:nboundary+shape[1]]
    #plt.figure(figsize=(40, 4))
    im2 = ax.imshow(wave.T, cmap=cmap_wave, origin='upper', aspect='auto', interpolation='none', vmin=SeisAmpMin, vmax=SeisAmpMax, extent=[xmin, xmax, zmax, zmin],)

    # Plot background velocity
    im1 = ax.imshow(velocity.T, cmap='jet', origin='upper', aspect='auto', alpha=alpha, interpolation='none', vmin=1.5, vmax=4.7, extent=[xmin, xmax, zmax, zmin],)


    

    if receiver_coordinate is not None:
        receivers_x = receiver_coordinate.T[0]/1000
        receivers_z = receiver_coordinate.T[1]/1000
        ax.scatter(receivers_x, receivers_z, color='cyan', marker='v', s=40, label='Receivers')

    if shot_coordinate is not None:
        shot_x = shot_coordinate[0]/1000
        shot_z = shot_coordinate[1]/1000
        ax.scatter(shot_x, shot_z, color='yellow', marker='*', s=100, label='Shot')
    

    

    plt.colorbar(im2, ax=ax, label='Amplitude', orientation='vertical')
    plt.colorbar(im1, ax=ax, label='Velocity (km/s)')
    
    ax.set_title(f'Wavefield at time {int((time_step*dt))} ms')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Z (km)')
    plt.tight_layout()
    if save == True :
        plt.savefig(f"{path}"+f"/wavefield_Shot_{int(1+(shot_coordinate[0]-4050)/100)}_{int((time_step*dt))+100000} ms.png")

    
    if show == True:
        plt.show()
    else :
        plt.close()





def create_video_from_images(image_folder, output_video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    
    # Urutkan gambar berdasarkan nama file
    images.sort()

    if not images:
        print("Tidak ada gambar ditemukan di folder yang ditentukan.")
        return

    # Baca gambar pertama untuk mendapatkan dimensi
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Tentukan codec dan objek VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk file .mp4
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Mulai membuat video dari {len(images)} gambar...")
    
    for image in images:
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Peringatan: Tidak dapat membaca gambar {image_path}. Melewatkan gambar ini.")

    video.release()
    
    temp_video = output_video_path.replace(".mp4", "_temp.mp4")

    # Re-encode ke H264
    os.system(
        f'ffmpeg -y -i "{output_video_path}" -vcodec libx264 "{temp_video}"'
    )

    # Hapus video lama
    os.remove(output_video_path)

    # Rename hasil encode menjadi nama lama
    os.rename(temp_video, output_video_path)

    print(f"Video berhasil dibuat di: {output_video_path}")