# app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Simulador de Campos e Gradiente", layout="wide")
st.title("🛰️ Campos escalares, vetoriais, gradiente e derivada direcional (interativo)")

# ========= Núcleo =========
def make_grid(xlim, n):
    x = np.linspace(-xlim, xlim, n)
    y = np.linspace(-xlim, xlim, n)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

def potential(kind, X, Y, origin, G, M, k_quad):
    dx = X - origin[0]
    dy = Y - origin[1]
    r = np.hypot(dx, dy)
    if kind == "Gravitacional (-GM/r)":
        # potencial específico (J/kg): V = -GM/r
        eps = 1e3
        V = -G * M / (r + eps)
    else:
        # Quadrático: V = 1/2 * k * r^2
        V = 0.5 * k_quad * (dx**2 + dy**2)
    return V

def grad_from_field(V, x, y):
    # gradV_y, gradV_x com base nos eixos (linhas->y, colunas->x)
    gradV_y, gradV_x = np.gradient(V, y[1]-y[0], x[1]-x[0])
    return gradV_x, gradV_y

def sample_grad_at(px, py, x, y, gradV_x, gradV_y):
    i = np.argmin(np.abs(x - px))
    j = np.argmin(np.abs(y - py))
    return np.array([gradV_x[j, i], gradV_y[j, i]])

def acceleration(kind, pos, origin, G, M, k_quad):
    dx = origin[0] - pos[0]
    dy = origin[1] - pos[1]
    if kind == "Gravitacional (-GM/r)":
        r = np.hypot(dx, dy) + 1e3
        a = (G * M / r**3) * np.array([dx, dy])  # = -∇V (V=-GM/r)
    else:
        # V = 1/2 k ||p-origin||^2  =>  ∇V = k (p-origin)  => a = -∇V
        a = -k_quad * (pos - origin)
    return a

def step_euler(kind, pos, vel, dt, origin, G, M, k_quad):
    vel = vel + acceleration(kind, pos, origin, G, M, k_quad) * dt
    pos = pos + vel * dt
    return pos, vel

def step_rk4(kind, pos, vel, dt, origin, G, M, k_quad):
    def deriv(p, v):
        a = acceleration(kind, p, origin, G, M, k_quad)
        return a
    # p' = v ; v' = a(p)
    k1_p = vel
    k1_v = deriv(pos, vel)

    k2_p = vel + 0.5*dt*k1_v
    k2_v = deriv(pos + 0.5*dt*k1_p, vel + 0.5*dt*k1_v)

    k3_p = vel + 0.5*dt*k2_v
    k3_v = deriv(pos + 0.5*dt*k2_p, vel + 0.5*dt*k2_v)

    k4_p = vel + dt*k3_v
    k4_v = deriv(pos + dt*k3_p, vel + dt*k3_v)

    pos_next = pos + (dt/6.0)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    vel_next = vel + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return pos_next, vel_next

# ========= Sidebar =========
with st.sidebar:
    st.header("Parâmetros do campo")
    campo_kind = st.selectbox("Campo escalar", ["Gravitacional (-GM/r)", "Quadrático (½ k r²)"])
    G = st.number_input("G (m³/kg/s²)", value=6.67430e-11, format="%.6e")
    M = st.number_input("Massa central M (kg)", value=5.972e24, format="%.3e")
    k_quad = st.number_input("k (para campo quadrático)", value=1e-12, format="%.3e")

    st.subheader("Espaço / Grade")
    xlim = st.number_input("Limite |x|=|y| (m)", value=6e7, step=1e6, format="%.0f")
    grid_n = st.slider("Resolução da grade", 10, 80, 26, 2)

    st.subheader("Derivada Direcional")
    ponto_x = st.number_input("Ponto x (m)", value=3e7, format="%.0f")
    ponto_y = st.number_input("Ponto y (m)", value=3e7, format="%.0f")
    theta_deg = st.slider("Ângulo θ (graus)", 0, 360, 45, 1)

    st.subheader("Exibição")
    show_streamlines = st.checkbox("Mostrar streamlines (linhas de campo)", True)
    show_colorbar = st.checkbox("Mostrar colorbar do escalar", True)

    st.subheader("Simulação orbital")
    x0 = st.number_input("x0 (m)", value=6e7, format="%.0f")
    y0 = st.number_input("y0 (m)", value=0.0, format="%.0f")
    vx0 = st.number_input("vx0 (m/s)", value=0.0, format="%.1f")
    vy0 = st.number_input("vy0 (m/s)", value=2000.0, format="%.1f")
    dt = st.number_input("Δt (s)", value=60.0, min_value=0.1, step=10.0, format="%.1f")
    steps = st.slider("Passos", 50, 4000, 800, 50)
    integrator_name = st.selectbox("Integrador", ["RK4 (estável)", "Euler (simples)"])
    show_bg_sim = st.checkbox("Fundo com escalar + campo na simulação", False)

origin = np.array([0.0, 0.0])
x, y, X, Y = make_grid(xlim, grid_n)
V = potential(campo_kind, X, Y, origin, G, M, k_quad)
gradV_x, gradV_y = grad_from_field(V, x, y)
Fx, Fy = -gradV_x, -gradV_y  # campo vetorial associado

theta = np.deg2rad(theta_deg)
v_hat = np.array([np.cos(theta), np.sin(theta)])
grad_p = sample_grad_at(ponto_x, ponto_y, x, y, gradV_x, gradV_y)
D_dir = float(grad_p @ v_hat)

# ========= Abas =========
tab_e, tab_v, tab_g, tab_d, tab_sim = st.tabs(
    ["🟦 Escalar", "🟧 Vetorial", "🟩 Gradiente", "🟪 Direcional", "🎬 Simulação"]
)

# --- Escalar ---
with tab_e:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
    ax.grid(True, alpha=0.2)
    ax.set_title("Campo escalar V(x,y) (contornos)")

    cf = ax.contourf(X, Y, V, levels=50, cmap='viridis', alpha=0.9)
    ax.contour(X, Y, V, levels=20, colors='k', linewidths=0.3, alpha=0.5)

    if show_colorbar:
        fig.colorbar(cf, ax=ax, label="Potencial V (J/kg)")

    st.pyplot(fig, clear_figure=True)

# --- Vetorial ---
with tab_v:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
    ax.grid(True, alpha=0.2)
    ax.set_title("Campo vetorial F = -∇V (quiver)")

    ax.quiver(X, Y, Fx, Fy, alpha=0.7)
    if show_streamlines:
        ax.streamplot(x, y, Fx, Fy, color='k', density=1.2, linewidth=0.8, arrowsize=1)

    st.pyplot(fig, clear_figure=True)

# --- Gradiente ---
with tab_g:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
    ax.grid(True, alpha=0.2)
    ax.set_title("Gradiente ∇V sobre as curvas de nível")

    cs = ax.contour(X, Y, V, levels=25, colors='k', linewidths=0.4, alpha=0.5)
    ax.quiver(X, Y, gradV_x, gradV_y, color='orange', alpha=0.7)

    # no ponto escolhido, mostre ∇V e v̂
    esc = xlim/6
    ax.quiver(ponto_x, ponto_y, grad_p[0]*esc, grad_p[1]*esc, color='orange', scale=1, scale_units='xy')
    ax.quiver(ponto_x, ponto_y, v_hat[0]*esc,  v_hat[1]*esc,  color='red',    scale=1, scale_units='xy')
    ax.plot([ponto_x], [ponto_y], 'bo')

    st.pyplot(fig, clear_figure=True)

# --- Direcional ---
with tab_d:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
        ax.grid(True, alpha=0.2)
        ax.set_title("Derivada direcional no ponto")

        cf = ax.contourf(X, Y, V, levels=40, cmap='viridis', alpha=0.85)
        ax.contour(X, Y, V, levels=20, colors='k', linewidths=0.3, alpha=0.5)
        if show_colorbar:
            fig.colorbar(cf, ax=ax, label="Potencial V (J/kg)")

        esc = xlim/6
        ax.quiver(ponto_x, ponto_y, grad_p[0]*esc, grad_p[1]*esc, color='orange', scale=1, scale_units='xy', label="∇V")
        ax.quiver(ponto_x, ponto_y, v_hat[0]*esc,  v_hat[1]*esc,  color='red',    scale=1, scale_units='xy', label="v̂")
        ax.legend(loc='upper right')
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.markdown(f"**D₍v̂₎V em ({ponto_x:.2e}, {ponto_y:.2e}) m:**")
        st.code(f"{D_dir:.3e} J/kg/m")
        thetas = np.linspace(0, 2*np.pi, 181)
        Dv = [float(grad_p @ np.array([np.cos(t), np.sin(t)])) for t in thetas]
        figp, axp = plt.subplots(figsize=(5,3.5))
        axp.plot(np.degrees(thetas), Dv)
        axp.set_xlabel("θ (graus)"); axp.set_ylabel("D_{v̂} V")
        axp.set_title("Variação de D_{v̂}V com θ")
        axp.grid(True, alpha=0.3)
        st.pyplot(figp, clear_figure=True)

# --- Simulação ---
with tab_sim:
    placeholder = st.empty()
    run = st.button("Iniciar simulação", type="primary")
    if run:
        pos = np.array([x0, y0], dtype=float)
        vel = np.array([vx0, vy0], dtype=float)
        xs, ys = [pos[0]], [pos[1]]

        stepper = step_rk4 if integrator_name.startswith("RK4") else step_euler
        for k in range(steps):
            pos, vel = stepper(campo_kind, pos, vel, dt, origin, G, M, k_quad)
            xs.append(pos[0]); ys.append(pos[1])

            fig2, ax2 = plt.subplots(figsize=(7,7))
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlim(-xlim, xlim); ax2.set_ylim(-xlim, xlim)
            ax2.grid(True, alpha=0.2)
            ax2.set_title(f"Órbita — passo {k+1}/{steps} | r = {np.hypot(pos[0], pos[1]):.2e} m")

            if show_bg_sim:
                cf2 = ax2.contourf(X, Y, V, levels=30, cmap='viridis', alpha=0.7)
                ax2.quiver(X, Y, Fx, Fy, alpha=0.35)

            ax2.plot(xs, ys, 'b-', lw=1, label="trajetória")
            ax2.plot(pos[0], pos[1], 'ro', label="corpo")
            ax2.legend(loc='upper right')

            placeholder.pyplot(fig2, clear_figure=True)
