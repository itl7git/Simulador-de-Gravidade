# app.py — Gradiente de um Campo Escalar (focado e didático)
# --------------------------------------------
# 5 presets prontos:
# 1) Plano Inclinado:     V = a x + b y            -> ∇V = (a, b) (constante)
# 2) Quadrático isótropo: V = 1/2 k (x^2 + y^2)    -> ∇V = (k x, k y)
# 3) Elíptico anisótropo: V = 1/2 (kx x^2 + ky y^2)-> ∇V = (kx x, ky y)
# 4) Sela (hiperbólico):  V = x^2 - y^2            -> ∇V = (2x, -2y)
# 5) Gaussiano (pico):    V = A exp(-x^2/2σx^2 - y^2/2σy^2)
#                          -> ∇V = V * (x/σx^2, y/σy^2) (aponta pro pico)
# --------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from skimage import filters, data
import matplotlib.colors as mcolors

st.set_page_config(page_title="Gradiente de um Campo Escalar", layout="wide")
st.title("∇V — Gradiente de um Campo Escalar (simulador didático)")

# ===================== Utilitários de grade e gradiente =====================
def make_grid(xlim, n):
    x = np.linspace(-xlim, xlim, n)
    y = np.linspace(-xlim, xlim, n)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def gradient(V, x, y):
    dy = y[1] - y[0]
    dx = x[1] - x[0]
    # Atenção: np.gradient retorna [dV/dy, dV/dx] na ordem dos eixos
    dV_dy, dV_dx = np.gradient(V, dy, dx)
    return dV_dx, dV_dy  # (grad_x, grad_y)

def sample_vec_at(px, py, x, y, U, V):
    i = int(np.argmin(np.abs(x - px)))
    j = int(np.argmin(np.abs(y - py)))
    return np.array([U[j, i], V[j, i]])

def unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

# ===================== Presets =====================
def get_presets():
    # Cada preset define: nome, fórmula LaTeX, função V(X,Y), domínios/valores padrão e ponto de destaque
    presets = []

    # 1) Plano Inclinado
    presets.append({
        "key": "plano",
        "nome": "Plano Inclinado",
        "latex": r"V(x,y) = a\,x + b\,y \quad\Rightarrow\quad \nabla V = (a,\; b)",
        "params": {"a": 1.0, "b": 0.5},
        "xlim": 5.0, "n": 41,
        "ponto": (1.5, -1.0),
        "V": lambda X, Y, p: p["a"]*X + p["b"]*Y
    })

    # 2) Quadrático isótropo
    presets.append({
        "key": "quad_iso",
        "nome": "Quadrático Isótropo",
        "latex": r"V(x,y) = \tfrac{1}{2}k(x^2 + y^2) \quad\Rightarrow\quad \nabla V = (k\,x,\; k\,y)",
        "params": {"k": 1.0},
        "xlim": 4.0, "n": 49,
        "ponto": (2.0, 1.0),
        "V": lambda X, Y, p: 0.5*p["k"]*(X**2 + Y**2)
    })

    # 3) Elíptico anisótropo
    presets.append({
        "key": "eliptico",
        "nome": "Quadrático Elíptico (anisótropo)",
        "latex": r"V(x,y) = \tfrac{1}{2}(k_x x^2 + k_y y^2) \Rightarrow \nabla V = (k_x x,\; k_y y)",
        "params": {"kx": 2.0, "ky": 0.6},
        "xlim": 4.0, "n": 49,
        "ponto": (1.5, 2.0),
        "V": lambda X, Y, p: 0.5*(p["kx"]*X**2 + p["ky"]*Y**2)
    })

    # 4) Sela
    presets.append({
        "key": "sela",
        "nome": "Sela (hiperbólico)",
        "latex": r"V(x,y) = x^2 - y^2 \quad\Rightarrow\quad \nabla V = (2x,\; -2y)",
        "params": {},
        "xlim": 4.0, "n": 49,
        "ponto": (2.0, 1.5),
        "V": lambda X, Y, p: (X**2 - Y**2)
    })

    # 5) Gaussiano (pico)
    presets.append({
        "key": "gauss",
        "nome": "Gaussiano (pico)",
        "latex": r"V(x,y) = A\,e^{-\frac{x^2}{2\sigma_x^2} - \frac{y^2}{2\sigma_y^2}}"
                 r"\;\Rightarrow\; \nabla V = V\left(\frac{x}{\sigma_x^2},\,\frac{y}{\sigma_y^2}\right)",
        "params": {"A": 1.0, "sigx": 1.4, "sigy": 0.8},
        "xlim": 4.0, "n": 59,
        "ponto": (1.0, 1.0),
        "V": lambda X, Y, p: p["A"]*np.exp(-(X**2)/(2*p["sigx"]**2) - (Y**2)/(2*p["sigy"]**2))
    })
    return presets

PRESETS = get_presets()
preset_names = [f"{i+1}) {pr['nome']}" for i, pr in enumerate(PRESETS)]
preset_idx = st.sidebar.radio("Escolha um exemplo (pré-definido)", list(range(len(PRESETS))),
                              format_func=lambda i: preset_names[i], index=0)

pr = PRESETS[preset_idx]

# ===================== Parâmetros do preset =====================
st.sidebar.subheader("Parâmetros do exemplo")
# Mostra os parâmetros (fixos por padrão), com opção de ajustar
params = dict(pr["params"])  # cópia

advanced = st.sidebar.checkbox("Ajustar parâmetros manualmente", value=False)
if advanced:
    if pr["key"] == "plano":
        params["a"] = st.sidebar.number_input("a", value=params["a"], step=0.1)
        params["b"] = st.sidebar.number_input("b", value=params["b"], step=0.1)
    elif pr["key"] == "quad_iso":
        params["k"] = st.sidebar.number_input("k", value=params["k"], step=0.1)
    elif pr["key"] == "eliptico":
        params["kx"] = st.sidebar.number_input("k_x", value=params["kx"], step=0.1)
        params["ky"] = st.sidebar.number_input("k_y", value=params["ky"], step=0.1)
    elif pr["key"] == "gauss":
        params["A"] = st.sidebar.number_input("A", value=params["A"], step=0.1)
        params["sigx"] = st.sidebar.number_input("σ_x", value=params["sigx"], step=0.1)
        params["sigy"] = st.sidebar.number_input("σ_y", value=params["sigy"], step=0.1)

# Grade e ponto de destaque
st.sidebar.subheader("Espaço e ponto de destaque")
xlim = st.sidebar.number_input("|x| = |y| (limite do plano)", value=float(pr["xlim"]), step=0.5)
n = st.sidebar.slider("Resolução da grade", min_value=25, max_value=101, step=2, value=pr["n"])
px = st.sidebar.number_input("x do ponto (destaque)", value=float(pr["ponto"][0]), step=0.1)
py = st.sidebar.number_input("y do ponto (destaque)", value=float(pr["ponto"][1]), step=0.1)

# Opções visuais
st.sidebar.subheader("Exibição")
show_quiver = st.sidebar.checkbox("Mostrar ∇V como setas (quiver)", value=True)
show_stream = st.sidebar.checkbox("Mostrar linhas de fluxo do ∇V (streamlines)", value=True)
show_colorbar = st.sidebar.checkbox("Mostrar colorbar", value=True)
skip = st.sidebar.slider("Espaçamento das setas (quiver)", 1, 5, 2)

# ===================== Cálculo =====================
x, y, X, Y = make_grid(xlim, n)
V = pr["V"](X, Y, params)

grad_x, grad_y = gradient(V, x, y)
Gnorm = np.sqrt(grad_x**2 + grad_y**2)

g_pt = sample_vec_at(px, py, x, y, grad_x, grad_y)
g_hat = unit(g_pt)
# vetor tangente (aprox.) à curva de nível local (ortogonal ao gradiente)
t_hat = unit(np.array([-g_pt[1], g_pt[0]]))

# ===================== ABAS =====================
tab_campo, tab_grad, tab_apl, tab_img = st.tabs(
    ["Campo V & ∇V", "Módulo |∇V|", "Aplicações reais", "🖼️ Imagem (Sobel)"]
)

# --- Aba 1: Campo + Gradiente ---
with tab_campo:
    colL, colR = st.columns([2, 1])
    with colL:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{pr['nome']} — V(x,y) e ∇V")
        # Contornos do escalar
        cf = ax.contourf(X, Y, V, levels=40, cmap="viridis", alpha=0.9)
        ax.contour(X, Y, V, levels=20, colors='k', linewidths=0.4, alpha=0.5)
        if show_colorbar:
            fig.colorbar(cf, ax=ax, label="V (escala arbitrária)")

        # Campo de gradiente (quiver/streamlines)
        if show_quiver:
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                      alpha=0.8, color="orange")

        if show_stream:
            ax.streamplot(x, y, grad_x, grad_y, color="white",
                          density=1.2, linewidth=0.8, arrowsize=1)

        # Ponto de destaque + vetores g (normal) e t (tangente)
        esc = xlim/5
        ax.plot([px], [py], 'ro', ms=6)
        ax.quiver(px, py, g_hat[0]*esc, g_hat[1]*esc, color='orange', scale=1, scale_units='xy', label="∇V (direção de maior aumento)")
        ax.quiver(px, py, t_hat[0]*esc, t_hat[1]*esc, color='cyan', scale=1, scale_units='xy', label="t (tangente à curva de nível)")
        ax.legend(loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with colR:
        st.subheader("Fórmula do exemplo")
        st.latex(pr["latex"])
        if pr["key"] == "plano":
            st.markdown("- **Gradiente constante**: todas as setas têm mesma direção e módulo.\n- As curvas de nível são **retas**; ∇V é **perpendicular** a elas.")
        elif pr["key"] == "quad_iso":
            st.markdown("- **Radialidade**: ∇V aponta para **fora**; módulo cresce com a distância.\n- Curvas de nível são **círculos**; ∇V é perpendicular aos círculos.")
        elif pr["key"] == "eliptico":
            st.markdown("- **Anisotropia**: alonga mais no eixo com **k maior**.\n- ∇V tem componente mais forte onde a curvatura é maior.")
        elif pr["key"] == "sela":
            st.markdown("- **Subida numa direção e descida na outra**.\n- ∇V muda de direção conforme quadrante; curvas de nível são **hipérboles**.")
        elif pr["key"] == "gauss":
            st.markdown("- **Pico central**: ∇V aponta para o **pico** (maior V).\n- Módulo máximo nas **encostas** do pico (onde V varia mais).")

# --- Aba 2: Módulo do gradiente ---
with tab_grad:
    col1, col2 = st.columns([2,1])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(7.5, 7.5))
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlim(-xlim, xlim); ax2.set_ylim(-xlim, xlim)
        ax2.grid(True, alpha=0.2)
        ax2.set_title("|∇V| — intensidade da subida máxima")
        c2 = ax2.contourf(X, Y, Gnorm, levels=40, cmap="magma", alpha=0.95)
        if show_colorbar:
            fig2.colorbar(c2, ax=ax2, label="|∇V| (taxa de variação máxima)")
        st.pyplot(fig2, clear_figure=True)
    with col2:
        st.write("- |∇V| mede **o quanto** V muda por unidade de distância.\n"
                 "- Regiões com **curvas de nível mais juntas** → |∇V| **maior**.\n"
                 "- No plano inclinado, |∇V| é **constante**.\n"
                 "- No quadrático, |∇V| cresce com a **distância ao centro**.\n"
                 "- No gaussiano, |∇V| é **máximo nas encostas**, não no pico.")
        st.markdown("**No ponto destacado**")
        st.code(f"∇V(px,py) = ({g_pt[0]:+.3f}, {g_pt[1]:+.3f})   |∇V| = {np.linalg.norm(g_pt):.3f}")

# --- Aba 3: Aplicações em Computação & Indústria ---
with tab_apl:
    st.header("Aplicações em Computação & Indústria (∇ de um escalar)")

    with st.expander("1) Bordas em imagens (Sobel/Canny)"):
        st.latex(r"G_x = I * K_x,\quad G_y = I * K_y,\quad |\nabla I|=\sqrt{G_x^2+G_y^2}")
        st.markdown("- **Intuição:** bordas são onde a intensidade muda rápido (**|∇I| alto**). "
                    "Base de detecção de bordas usada em câmeras, inspeção de qualidade e digitalização.")

    with st.expander("2) SIFT / HOG (descritores por orientação do gradiente)"):
        st.latex(r"\theta = \mathrm{atan2}(G_y, G_x)")
        st.markdown("- **Ideia:** construir histogramas de **orientação do gradiente** por células. "
                    "Usado em detecção/descrição de características, OCR e visão embarcada.")

    with st.expander("3) Optical Flow (Lucas–Kanade)"):
        st.latex(r"\begin{bmatrix}I_x & I_y\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix} \approx -I_t")
        st.markdown("- **Ideia:** movimentos pequenos resolvem sistema local com **∇I** e derivada temporal. "
                    "Aplicado em ADAS/autônomos e estabilização de vídeo.")

    with st.expander("4) Seam Carving (redimensionamento consciente de conteúdo)"):
        st.latex(r"E(x,y)=|\nabla I(x,y)|")
        st.markdown("- **Ideia:** remover/insert seams de **menor energia** para preservar conteúdo. "
                    "Usado em edição de imagens e UIs adaptativas.")

    with st.expander("5) Blending no domínio do gradiente (Poisson blending)"):
        st.latex(r"\Delta f = \nabla \cdot \mathbf{v} \quad (\text{reconstruir imagem a partir de gradientes})")
        st.markdown("- **Ideia:** colar objetos preservando padrões de gradiente. "
                    "Ferramentas como Photoshop/GIMP usam variantes desse princípio.")

    with st.expander("6) Normals de height maps (gráficos/jogos)"):
        st.latex(r"\mathbf{n} \propto \big(-\tfrac{\partial z}{\partial x},\ -\tfrac{\partial z}{\partial y},\ 1\big)")
        st.markdown("- **Ideia:** a partir de um **height field**, ∇z dá normais para iluminação realista. "
                    "Presente em engines 3D e VFX.")

    with st.expander("7) SDF/TSDF em reconstrução 3D"):
        st.markdown("- **Ideia:** o **gradiente do SDF** fornece **normais** e orientações superficiais. "
                    "Base para mapeamento 3D (AR/VR), ray-marching e ICP em reconstrução.")
        
    with st.expander("8) Otimização & ML (backprop / explicabilidade)"):
        st.latex(r"\theta \leftarrow \theta - \eta \nabla L(\theta)")
        st.markdown("- **Ideia:** treinar modelos = descer o **gradiente do loss**. "
                    "**Integrated Gradients** para atribuição de importância de features.")
        
# --- Aba 4: Imagem (Sobel) ---
with tab_img:
    st.subheader("Gradiente em imagens: |∇I| (Sobel)")
    left, right = st.columns([1,1])

    with left:
        file = st.file_uploader("Envie uma imagem (png/jpg/jpeg)", type=["png","jpg","jpeg"])
        use_sample = st.checkbox("Usar imagem de exemplo (camera)", value=True)
        sigma = st.slider("Suavização (σ) antes do Sobel", 0.0, 3.0, 1.0, 0.1)
        show_ori = st.checkbox("Mostrar orientação do gradiente (matiz)", value=False)
        thr = st.slider("Limiar para 'bordas' sobre |∇I|", 0.0, 1.0, 0.30, 0.01)

    # --- carregar imagem ---
    if use_sample:
        pil = Image.fromarray(data.camera())
    else:
        if not file:
            st.info("Envie uma imagem ou ative 'Usar imagem de exemplo'.")
            st.stop()
        pil = Image.open(file).convert("RGB")

    gray = np.asarray(pil.convert("L"), dtype=np.float32) / 255.0

    # --- suavização opcional ---
    if sigma > 0:
        from skimage.filters import gaussian
        gray_s = gaussian(gray, sigma=sigma, preserve_range=True)
    else:
        gray_s = gray

    # --- derivadas Sobel (horizontal/vertical), magnitude e orientação ---
    Gx = filters.sobel_h(gray_s)
    Gy = filters.sobel_v(gray_s)
    mag = np.hypot(Gx, Gy)
    mag /= (mag.max() + 1e-8)
    ori = (np.arctan2(Gy, Gx) + np.pi) / (2*np.pi)  # 0..1 (mapeável para matiz)

    with right:
        # imagem original
        fig0, ax0 = plt.subplots()
        ax0.imshow(gray, cmap="gray")
        ax0.set_title("Imagem (tons de cinza)"); ax0.axis("off")
        st.pyplot(fig0, clear_figure=True)

        # magnitude do gradiente
        fig1, ax1 = plt.subplots()
        ax1.imshow(mag, cmap="magma")
        ax1.set_title("|∇I| (Sobel)"); ax1.axis("off")
        st.pyplot(fig1, clear_figure=True)

        # orientação colorida (HSV: H=ori, V=mag)
        if show_ori:
            HSV = np.zeros((*ori.shape, 3), dtype=np.float32)
            HSV[..., 0] = ori         # matiz = direção do gradiente
            HSV[..., 1] = 1.0         # saturação
            HSV[..., 2] = mag         # valor = intensidade
            RGB = mcolors.hsv_to_rgb(HSV)
            fig2, ax2 = plt.subplots()
            ax2.imshow(RGB)
            ax2.set_title("Orientação (matiz) + |∇I| (valor)")
            ax2.axis("off")
            st.pyplot(fig2, clear_figure=True)

        # bordas por limiar na magnitude
        edges = (mag >= thr).astype(float)
        fig3, ax3 = plt.subplots()
        ax3.imshow(gray, cmap="gray")
        ax3.imshow(np.ma.masked_where(edges == 0, edges), alpha=0.7, cmap="cool")
        ax3.set_title(f"Bordas (|∇I| ≥ {thr:.2f}) sobre a imagem")
        ax3.axis("off")
        st.pyplot(fig3, clear_figure=True)

    st.caption("Dica didática: bordas são regiões onde a intensidade muda rápido ⇒ |∇I| alto. σ controla ruído; limiar destaca contornos mais fortes.")
