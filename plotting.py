import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Set global font to Times New Roman ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- Data from your mesh convergence table ---
dof_cone   = [11602, 22963, 68869, 267037]
time_cone  = [14.0, 52.0, 183.0, 1240.0]        # seconds
error_cone = [7.69e-04, 7.07e-04, 2.23e-04, 0.00e-04]  # %

dof_rectangle = [10926,21096,64375,248648]
time_rectangle =[20,45,181,1069]
error_rectangle = [2.30E-03,1.36E-03,1.16E-04,0.00E+00]

dof_pillar_recessed = [676, 2059, 3900, 11315, 22438, 68145, 262952]
time_pillar_recessed = [5,8,13,2.70E+01,6.50E+01,2.51E+02,1511]
error_pillar_recessed = [4.75E-03,2.02E-03,1.85E-03,1.32E-03,8.73E-04,1.57319E-05,0]

dof_triangle_recessed = [10726,21717,66077,255341]
time_triangle_recessed = [2.00E+01,4.60E+01,2.07E+02,1216]
error_triangle_recessed = [1.55E-03,1.10E-03,8.12E-05,0.00E+00]

dof_square_recessed = [11626, 22759, 68863, 265997]
time_square_recessed = [2.10E+01,3.80E+01,1.53E+02,1036]    
error_square_recessed = [0.001701426,0.000978418,7.75302E-05,0.000000000] 

dof_square_additive = [11837,23237,70780,274663]
time_square_additive = [1.70E+01,2.80E+01,1.71E+02,1402]
error_square_additive = [0.001014427,0.00085764,9.75146E-05,0.00E+00]

dof_cylindrical_pillars = [12116,23621, 71276, 278469]
time_cylindrical_pillars = [1.50E+01,3.70E+01,1.50E+02,1360]
error_cylindrical_pillars = [2.23E-03,1.60E-03,4.70E-04,0.00E+00]

fig, ax1 = plt.subplots(figsize=(8, 6))

# Left axis: relative error
color1 = 'darkred'
ax1.set_xlabel('DoF', fontsize=14)
ax1.set_ylabel('Relative error/%', color=color1, fontsize=14)
ax1.plot(dof_cylindrical_pillars, error_cylindrical_pillars, marker='o', markersize=10, color=color1,
          linestyle='-', linewidth=1.5, markerfacecolor=color1,
          markeredgecolor='black', label='Relative error')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')

# Format y-axis tick labels to 3 significant figures
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))

# Right axis: computation time
ax2 = ax1.twinx()
color2 = 'navy'
ax2.set_ylabel('Time/s', color=color2, fontsize=14)
ax2.plot(dof_cylindrical_pillars, time_cylindrical_pillars , marker='s', markersize=9, color=color2,
          linestyle='--', linewidth=1.5, markerfacecolor=color2,
          markeredgecolor='black', label='Time')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.savefig('mesh_convergence.png', dpi=300)
plt.show()
