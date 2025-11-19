import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) DC Motor parametreleri
# ---------------------------
R = 1.0
L = 0.5
Kt = 0.01
Kb = 0.01
J = 0.01
B = 0.001
Vmax = 24.0  # Besleme gerilimi

# ---------------------------
# 2) Üyelik fonksiyonları
# ---------------------------

def triangular(x, a, b, c):
    """
    Basit üçgensel üyelik fonksiyonu.
    a: sol uç, b: tepe, c: sağ uç
    """
    x = np.asarray(x, dtype=float)
    mu = np.zeros_like(x, dtype=float)

    # Sol kenar (a -> b)
    left = (a < x) & (x <= b)
    mu[left] = (x[left] - a) / (b - a + 1e-12)

    # Sağ kenar (b -> c)
    right = (b < x) & (x < c)
    mu[right] = (c - x[right]) / (c - b + 1e-12)

    # Tepe noktası
    mu[x == b] = 1.0

    # a'nın solunda ve c'nin sağında zaten 0
    return mu

# --- Hata (e) için 5 üyelik fonksiyonu ---
#  Düzenlendi: Motorun referansı 100 rad/s civarı olduğu için
#  hata aralığını [-150, 150] bandına çektik ve merkezdeki Z bölgesini daralttık.
e_NB = (-150, -100, -50)   # Negatif Büyük
e_NS = (-100, -50, 0)      # Negatif Küçük
e_Z  = (-10, 0, 10)        # Sıfır
e_PS = (0, 50, 100)        # Pozitif Küçük
e_PB = (50, 100, 150)      # Pozitif Büyük

# --- Hata değişimi (de) için 3 üyelik fonksiyonu ---
#  Düzenlendi: de önceki koda göre çok genişti ([-300, 300]).
#  Biz motor hızının değişiminin daha küçük olacağını varsayıp daralttık.
de_N = (-50, -25, 0)       # Negatif (azalan hata)
de_Z = (-10, 0, 10)        # Sıfır (hata değişmiyor)
de_P = (0, 25, 50)         # Pozitif (artan hata)

# --- Çıkış (u) üyelikleri ---
#  Düzenlendi: Çıkış için üçgenleri Vmax etrafında daha gerçekçi bir şekilde tanımladık.
#  Aşırı doygunluğun önüne geçmek için uçları biraz yumuşak tuttuk.
u_N = (-Vmax, -0.7*Vmax, 0)     # Negatif kontrol
u_Z = (-4, 0, 4)                # Sıfır civarı küçük düzeltme
u_P = (0, 0.7*Vmax, Vmax)       # Pozitif kontrol

# --- 5x3 kural tablosu (e x de) ---
# e: [NB, NS, Z, PS, PB]
# de: [N, Z, P]
# TABLO DÜZENLENDİ:
#  - Büyük hata ve pozitif de durumunda (hata büyüyor) daha agresif kontrol (P)
#  - Hata sıfıra yakın ve de küçükken Z ağırlıklı
rule_table = [
    ['N', 'N', 'N'],   # NB
    ['N', 'N', 'Z'],   # NS
    ['N', 'Z', 'P'],   # Z
    ['Z', 'P', 'P'],   # PS
    ['P', 'P', 'P']    # PB
]

output_mfs = {'N': u_N, 'Z': u_Z, 'P': u_P}

# --- Fuzzification ---
def fuzzify_e_de(e, de):
    # Hata için üyelikler
    mu_e = {
        'NB': triangular([e], *e_NB)[0],
        'NS': triangular([e], *e_NS)[0],
        'Z':  triangular([e], *e_Z)[0],
        'PS': triangular([e], *e_PS)[0],
        'PB': triangular([e], *e_PB)[0],
    }

    # Hata değişimi için üyelikler
    mu_de = {
        'N': triangular([de], *de_N)[0],
        'Z': triangular([de], *de_Z)[0],
        'P': triangular([de], *de_P)[0]
    }

    return mu_e, mu_de

# --- Mamdani Defuzzification ---
def mamdani_defuzz(e, de, u_disc=np.linspace(-Vmax, Vmax, 1001)):
    """
    Mamdani çıkarımı + ağırlık merkezi ile defuzzification.
    """
    mu_e, mu_de = fuzzify_e_de(e, de)
    aggregated = np.zeros_like(u_disc)

    e_labels = ['NB', 'NS', 'Z', 'PS', 'PB']
    de_labels = ['N', 'Z', 'P']

    for i_e, e_lab in enumerate(e_labels):
        for j_de, de_lab in enumerate(de_labels):
            fire = min(mu_e[e_lab], mu_de[de_lab])
            if fire <= 0:
                continue
            a, b, c = output_mfs[rule_table[i_e][j_de]]
            mu_out = triangular(u_disc, a, b, c)
            aggregated = np.maximum(aggregated, np.minimum(mu_out, fire))

    num = np.sum(u_disc * aggregated)
    den = np.sum(aggregated)

    if den == 0:
        u = 0.0
    else:
        u = num / den

    # EK DÜZELTME: Çıkışı fiziksel sınırlara göre saturate ediyoruz.
    u = float(np.clip(u, -Vmax, Vmax))
    return u

# ---------------------------
# 3) DC Motor dinamiği
# ---------------------------
def motor_derivatives(x, u, TL=0.0):
    """
    x = [i, w]
    i : armatür akımı
    w : açısal hız (rad/s)
    u : uygulanan gerilim
    TL: yük momenti
    """
    i, w = x
    di = (-R * i - Kb * w + u) / L
    dw = (-B * w + Kt * i - TL) / J
    return np.array([di, dw], dtype=float)

def rk4_step(x, u, dt, TL=0.0):
    """
    4. dereceden Runge-Kutta integrasyon adımı
    """
    k1 = motor_derivatives(x, u, TL)
    k2 = motor_derivatives(x + 0.5 * dt * k1, u, TL)
    k3 = motor_derivatives(x + 0.5 * dt * k2, u, TL)
    k4 = motor_derivatives(x + dt * k3, u, TL)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---------------------------
# 4) Kapalı çevrim simülasyonu
# ---------------------------
def simulate(ref_func, T=5.0, dt=0.001, x0=None, TL_func=None):
    """
    ref_func(t): referans hız fonksiyonu
    TL_func(t): zamanla değişen yük momenti (opsiyonel)
    """
    if x0 is None:
        x = np.array([0.0, 0.0], dtype=float)
    else:
        x = np.array(x0, dtype=float)

    t = np.arange(0, T + dt, dt)
    N = len(t)

    i_hist   = np.zeros(N)
    w_hist   = np.zeros(N)
    u_hist   = np.zeros(N)
    e_hist   = np.zeros(N)
    ref_hist = np.zeros(N)

    prev_e = 0.0

    for k in range(N):
        tk = t[k]
        ref = ref_func(tk)
        ref_hist[k] = ref

        e = ref - x[1]          # hız hatası
        de = e - prev_e         # hata değişimi (basit fark)
        prev_e = e

        u = mamdani_defuzz(e, de)

        if TL_func is not None:
            TL = TL_func(tk)
        else:
            TL = 0.0

        x = rk4_step(x, u, dt, TL)

        i_hist[k] = x[0]
        w_hist[k] = x[1]
        u_hist[k] = u
        e_hist[k] = e

    return t, w_hist, u_hist, e_hist, ref_hist, i_hist

# ---------------------------
# 5) Örnek simülasyon
# ---------------------------
if __name__ == "__main__":
    # Referans fonksiyonu:
    # 0–0.2 s arası rampa, sonra sabit 100 rad/s
    ref_val = 100.0  # rad/s

    def ref(t):
        if t < 0.2:
            return ref_val * (t / 0.2)
        else:
            return ref_val

    # Örnek yük momenti: 2. saniyeden sonra küçük bir yük eklensin
    def TL_func(t):
        return 0.002 if t >= 2.0 else 0.0

    t, w, u, e, ref_sig, i = simulate(ref, T=5.0, dt=0.001, TL_func=TL_func)

    # Hız grafiği
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(t, ref_sig, '--', label='Referans (rad/s)')
    plt.plot(t, w, label='Motor Hızı (rad/s)')
    plt.ylabel('Hız (rad/s)')
    plt.legend()
    plt.grid(True)

    # Gerilim grafiği
    plt.subplot(4, 1, 2)
    plt.plot(t, u)
    plt.ylabel('Gerilim u (V)')
    plt.grid(True)

    # Hata grafiği
    plt.subplot(4, 1, 3)
    plt.plot(t, e)
    plt.ylabel('Hata e')
    plt.grid(True)

    # Akım grafiği
    plt.subplot(4, 1, 4)
    plt.plot(t, i)
    plt.ylabel('Akım i (A)')
    plt.xlabel('Zaman (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 6) TÜM ÜYELİK FONKSİYONLARININ GRAFİKLERİ
    # ============================================================

    # a) Hata (e) üyelik fonksiyonları
    e_vals = np.linspace(-150, 150, 600)
    mu_NB = triangular(e_vals, *e_NB)
    mu_NS = triangular(e_vals, *e_NS)
    mu_Z  = triangular(e_vals, *e_Z)
    mu_PS = triangular(e_vals, *e_PS)
    mu_PB = triangular(e_vals, *e_PB)

    plt.figure(figsize=(8, 4))
    plt.plot(e_vals, mu_NB, label='NB (Negatif Büyük)')
    plt.plot(e_vals, mu_NS, label='NS (Negatif Küçük)')
    plt.plot(e_vals, mu_Z,  label='Z (Sıfır)')
    plt.plot(e_vals, mu_PS, label='PS (Pozitif Küçük)')
    plt.plot(e_vals, mu_PB, label='PB (Pozitif Büyük)')
    plt.title('Hata (e) Üyelik Fonksiyonları')
    plt.xlabel('Hata (e)')
    plt.ylabel('Üyelik Derecesi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # b) Hata değişimi (de) üyelik fonksiyonları
    de_vals = np.linspace(-50, 50, 400)
    mu_de_N = triangular(de_vals, *de_N)
    mu_de_Z = triangular(de_vals, *de_Z)
    mu_de_P = triangular(de_vals, *de_P)

    plt.figure(figsize=(8, 4))
    plt.plot(de_vals, mu_de_N, label='N (Negatif)')
    plt.plot(de_vals, mu_de_Z, label='Z (Sıfır)')
    plt.plot(de_vals, mu_de_P, label='P (Pozitif)')
    plt.title('Hata Değişimi (de) Üyelik Fonksiyonları')
    plt.xlabel('Hata Değişimi (de)')
    plt.ylabel('Üyelik Derecesi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # c) Çıkış (u) üyelik fonksiyonları
    u_vals = np.linspace(-Vmax, Vmax, 400)
    mu_u_N = triangular(u_vals, *u_N)
    mu_u_Z = triangular(u_vals, *u_Z)
    mu_u_P = triangular(u_vals, *u_P)

    plt.figure(figsize=(8, 4))
    plt.plot(u_vals, mu_u_N, label='N (Negatif)')
    plt.plot(u_vals, mu_u_Z, label='Z (Sıfır)')
    plt.plot(u_vals, mu_u_P, label='P (Pozitif)')
    plt.title('Çıkış (u) Üyelik Fonksiyonları')
    plt.xlabel('Çıkış Gerilimi (u)')
    plt.ylabel('Üyelik Derecesi')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
