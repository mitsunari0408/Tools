import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import cg
import time

# ==========================================
# 1. SETUP
# ==========================================
# Resolution
dx = dy = dz = 0.2e-6 # distance between nodes

# Geometry
L_Si   = 10.0e-6
L_SiGe = 1.0e-6
L_Ge   = 1.0e-6 

Lx = L_Si + L_SiGe + L_Ge 
Ly = 5.0e-6
Lz = 15.0e-6

Nx = int(Lx / dx)
Ny = int(Ly / dy)
Nz = int(Lz / dz)

# Force Odd Grid for Center Pixel
if Ny % 2 == 0: Ny += 1
if Nz % 2 == 0: Nz += 1

N = Nx * Ny * Nz
print(f"New Grid: {Nx}x{Ny}x{Nz} ({N/1e6:.2f} M nodes)")

# ==========================================
# 2. MATERIAL PROPERTIES
# ==========================================
k_map     = np.zeros((Nx, Ny, Nz))
R_map     = np.zeros((Nx, Ny, Nz))
d_abs_map = np.zeros((Nx, Ny, Nz))

idx_1 = int(L_Si / dx)
idx_2 = int((L_Si + L_SiGe) / dx)

# Silicon
k_map[:idx_1, :, :] = 148.0;   R_map[:idx_1, :, :] = 0.348;   d_abs_map[:idx_1, :, :] = 3.0e-6
# SiGe
k_map[idx_1:idx_2, :, :] = 10.0;  R_map[idx_1:idx_2, :, :] = 0.406; d_abs_map[idx_1:idx_2, :, :] = 0.25e-6
# Germanium
k_map[idx_2:, :, :] = 56.0;    R_map[idx_2:, :, :] = 0.485;   d_abs_map[idx_2:, :, :] = 0.04e-6

# ==========================================
# 3. LASER SOURCE
# ==========================================
P_laser = 5e-4
w0 = 0.385e-6
xc = 5e-6 # laser incident center

x = np.linspace(0, Lx, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

r_sq = (X - xc)**2 + Y**2
intensity = np.exp(-2 * r_sq / w0**2)
area_norm = np.sum(intensity[:, :, 0]) * dx * dy
I0 = intensity * (P_laser / area_norm)

q_3D = I0 * (1 - R_map) * (1.0/d_abs_map) * np.exp(-Z/d_abs_map)
b = -q_3D.flatten()

# ==========================================
# 4. MATRIX BUILD (With Harmonic Mean for Interfaces)
# ==========================================
print("Building Matrix (Harmonic Mean k)...")
t0 = time.time()

nodes = np.arange(N).reshape((Nx, Ny, Nz))
main_diag = np.zeros(N)
rows, cols, data = [], [], []

inv_dx2 = 1.0/dx**2; inv_dy2 = 1.0/dy**2; inv_dz2 = 1.0/dz**2

def add_couplings_harmonic(slice_L, slice_R, k_L, k_R, inv_d2):
  
    u = slice_L.flatten()
    v = slice_R.flatten()
    
    # 1. Get k values for both sides of the interface
    k1 = k_L.flatten()
    k2 = k_R.flatten()
    
    # 2. Calculate Harmonic Mean
    # add a tiny epsilon (1e-20) to avoid divide by zero errors if k=0
    k_eff = (2 * k1 * k2) / (k1 + k2 + 1e-20)
    
    # 3. Calculate Conductance
    val = k_eff * inv_d2
    
    # 4. Build Matrix
    np.add.at(main_diag, u, -val)
    np.add.at(main_diag, v, -val)
    rows.append(u); cols.append(v); data.append(val)
    rows.append(v); cols.append(u); data.append(val)

# X-Direction Connections (Harmonic Mean)
add_couplings_harmonic(nodes[:-1,:,:], nodes[1:,:,:], 
                       k_map[:-1,:,:], k_map[1:,:,:], inv_dx2)

# Y-Direction Connections (Harmonic Mean)
add_couplings_harmonic(nodes[:,:-1,:], nodes[:,1:,:], 
                       k_map[:,:-1,:], k_map[:,1:,:], inv_dy2)

# Z-Direction Connections (Harmonic Mean)
add_couplings_harmonic(nodes[:,:,:-1], nodes[:,:,1:], 
                       k_map[:,:,:-1], k_map[:,:,1:], inv_dz2)

# BOUNDARY CONDITIONS (Standard Infinite Wafer)
# Right Edge (Choose: True for Infinite, False for Air)
T_base = 300.0
penalty = 1e15
mask = np.zeros((Nx, Ny, Nz), dtype=bool)

mask[0, :, :]   = True  # Left
mask[:, :, -1]  = True  # Bottom
mask[:, 0, :]   = False  # Front
mask[:, -1, :]  = False  # Back
mask[-1, :, :]  = False # right
mask[:, :, 0]   = False # top

flat_mask = mask.flatten()
main_diag[flat_mask] += penalty
b[flat_mask] += penalty * T_base

rows.append(np.arange(N)); cols.append(np.arange(N)); data.append(main_diag)
rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
A = csc_matrix((data, (rows, cols)), shape=(N, N))

print(f"Matrix Built in {time.time()-t0:.2f}s")

# ==========================================
# 5. SOLVE & PLOT
# ==========================================
print("Solving...")
t1 = time.time()
T_flat, info = cg(A, b, rtol=1e-5) 
print(f"Solved in {time.time()-t1:.2f}s")

# 5.1 CALCULATE RAMAN TEMPERATURE
# a. Get the 3D Temperature Field
T_3D = T_flat.reshape((Nx, Ny, Nz))
T_rise = T_3D - T_base

# 5.1.1. Calculate the Weighted Average
numerator   = np.sum(T_rise * q_3D)
denominator = np.sum(q_3D)

if denominator == 0:
    T_raman_val = 0
else:
    T_raman_val = numerator / denominator

# 5.2.2. Print the Result
print(f"Raman Temperature:   {T_raman_val:.2f} K (Rise)")
print(f"-------------------------------------------")

# 5.3 Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Surface
im1 = ax1.imshow(T_rise[:,:,0].T, origin='lower', extent=[0,Lx*1e6,-Ly/2*1e6,Ly/2*1e6], cmap='inferno')
fig.colorbar(im1, ax=ax1, label='Delta T (K)')
ax1.set_title('Surface Heat Profile')
ax1.set_xlabel('X (um)'); ax1.set_ylabel('Y (um)')
ax1.axvline(x=L_Si*1e6, c='w', ls='--')
ax1.axvline(x=(L_Si+L_SiGe)*1e6, c='w', ls='--')

# Plot 2: Depth
im2 = ax2.imshow(T_rise[:, int(Ny/2), :].T, origin='upper', extent=[0,Lx*1e6,Lz*1e6,0], cmap='inferno', aspect='auto')
fig.colorbar(im2, ax=ax2, label='Delta T (K)')
ax2.set_title('Depth Heat Profile')
ax2.set_xlabel('X (um)'); ax2.set_ylabel('Depth (um)')
ax2.axvline(x=L_Si*1e6, c='w', ls='--')
ax2.axvline(x=(L_Si+L_SiGe)*1e6, c='w', ls='--')

plt.tight_layout()

# --- FIGURE WINDOW 2: CENTERLINE PROFILE ---
fig2, ax3 = plt.subplots(figsize=(10, 6))
fig2.canvas.manager.set_window_title('Figure 2: Centerline Profile')

# Get Data at Center Y
mid_y = int(Ny / 2)
x_um = x * 1e6
T_line = T_rise[:, mid_y, 0]        # Temperature
I_line = intensity[:, mid_y, 0]     # Laser

# Plot Temperature (Left Axis)
color = 'tab:red'
ax3.set_xlabel('X Position (um)', fontsize=12)
ax3.set_ylabel('Temperature Rise (K)', color=color, fontsize=12, fontweight='bold')
line1 = ax3.plot(x_um, T_line, color=color, linewidth=2.5, label='Temperature')
ax3.tick_params(axis='y', labelcolor=color)
ax3.grid(True, linestyle=':', alpha=0.6)

# Plot Laser Intensity (Right Axis)
ax3b = ax3.twinx()
color = 'tab:blue'
ax3b.set_ylabel('Laser Intensity (Normalized)', color=color, fontsize=12, fontweight='bold')
line2 = ax3b.plot(x_um, I_line, color=color, linestyle='--', alpha=0.6, label='Laser Profile')
ax3b.tick_params(axis='y', labelcolor=color)

# Combine Legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left')

# Mark Interfaces
ax3.axvline(x=L_Si*1e6, color='black', alpha=0.3)
ax3.axvline(x=(L_Si+L_SiGe)*1e6, color='black', alpha=0.3)

# Add Region Labels
y_max = np.max(T_line)
ax3.text(L_Si*1e6 - 1, y_max*0.95, "Silicon (Bulk)", ha='right', color='gray')
ax3.text((L_Si + L_SiGe/2)*1e6, y_max*0.95, "SiGe", ha='center', fontweight='bold')
ax3.text((L_Si + L_SiGe)*1e6 +0.25, y_max*0.95, "Germanium", ha='left', color='gray')

ax3.set_title('Centerline Temperature Profile (y=0)', fontsize=14)
ax3.set_xlim(0, 16)

plt.tight_layout()
plt.show()