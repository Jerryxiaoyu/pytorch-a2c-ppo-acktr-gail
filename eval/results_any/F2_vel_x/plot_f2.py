import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt



# plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
#                                'Lucida Grande', 'Verdana']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble'] = [
#     r'\usepackage{tgheros}',    # helvetica font
#     r'\usepackage{sansmath}',   # math-font matching  helvetica
#     r'\sansmath'                # actually tell tex to use it!
#     r'\usepackage{siunitx}',    # micro symbols
#     r'\sisetup{detect-all}',    # force siunitx to use the fonts
# ]
#plt.rc('font', family='serif', serif='DejaVu Sans')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

# width as measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

x = np.arange(0.0, 3*np.pi , 0.1)
plt.plot(x, np.sin(x))

ax.set_ylabel('Some Metric (in unit)')
ax.set_xlabel('Something (in unit)')
ax.set_xlim(0, 3*np.pi)

fig.set_size_inches(width, height)
fig.savefig('plot.pdf')