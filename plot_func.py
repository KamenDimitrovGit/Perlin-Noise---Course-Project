import matplotlib.pyplot as plt
import numpy as np


def plot_points(x, y, c, txt=None, ):
    for xi, yi in zip(x, y):
        if txt:
            plt.text(xi, yi, s = f"({txt})")
        else:
            plt.text(xi, yi, s = f"({xi}, {yi})")
    plt.plot(x, y, c)

    
def def_axes(gridlabels=None, xlim=None, ylim=None, ax_labels=None, ticks = None, fontsize = "x-large"):
    ax = plt.gca()
    
    
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    
    if ylim:
        ax.set_ylim(ylim[0], ylim[1]);
    
    if ax_labels:
        plt.xlabel(ax_labels[0]);
        plt.ylabel(ax_labels[1])
    
    if ticks:
        ax.set_xticks(ticks[0])
        ax.set_yticks(ticks[1]);
    
    
    if gridlabels:
        labels = gridlabels
        ax.legend(labels)
    
    ax.grid(ls = '--')
    ax.set_aspect("equal")
    

def plot_square(s):
    plt.plot([-s,s,s,-s,-s],[s,s,-s,-s,s], "g-")
    


# Some useful functions
def draw_brace(ax, xspan, text):
    """Draws an annotated brace on the axes."""
    
    np.seterr(divide = 'ignore') 
    
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()

    yspan = (ymax - ymin) * 2
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:resolution//2+1]
    y_half_brace = (1/(1.-np.exp(-beta*(x_half-x_half[0])))
                    - 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = 0.30 + ymin + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., ymin+.3*yspan, text, fontsize = "x-large", ha='center', va='bottom')


# plt.figure(figsize = (15,15))

# pf.plot_points([0,1], [0,0], c = "ro", txt="$x_0$")
# pf.plot_points([0.42], [0], c = "bo", txt="$x_i$")
# pf.plot_points([0.42], [-0.3], c = "bo", txt="$x_i$")
# pf.plot_points([1], [0], c = "ro", txt="$x_1$")
# plt.plot([0.42,0.42],[-0.3,0], "g--")

# pf.def_axes(xlim=[-0.1, 1.1], ylim=[-0.35, 0.1], ax_labels=["x","y"], ticks=[range(2),range(1)])

# ax = plt.gca()
# draw_brace(ax, (0, 0.42), "$x_0 - x_i$")

# plt.show()

def run_perlin_noise(n, octave=1, permutation_table=None):
    
    pic = np.zeros((n, n, 1))    
    for x in range(n):
        for y in range(n):
            point = noise_2d(x, y, octave, permutation_table)
            
            point += 1
            point /= 2
            pic[x,y] = point
    return pic

def plot_first_noise(pic):
    
    f, ([ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]) = plt.subplots(3, 3, figsize = (15, 15))

    for axi in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        axi.axis("Off")
    
    ax1.set_title("n = 128, octave = 1")
    ax1.imshow(pic[0], cmap = "gray")

    ax2.set_title("n = 128, octave = 2")
    ax2.imshow(pic[1], cmap = "gray")

    ax3.set_title("n = 128, octave = 4")
    ax3.imshow(pic[2], cmap = "gray")

    ax4.set_title("n = 128, octave = 8")
    ax4.imshow(pic[3], cmap = "gray")

    ax5.set_title("n = 128, octave = 16")
    ax5.imshow(pic[4], cmap = "gray")

    ax6.set_title("n = 128, octave = 32")
    ax6.imshow(pic[5], cmap = "gray")

    ax7.set_title("n = 128, octave = 64")
    ax7.imshow(pic[6], cmap = "gray")

    ax8.set_title("Sum(Pic0 : Pic 4)")
    ax8.imshow(pic[7], cmap = "gray")

    ax9.set_title("Sum(Pic1 : Pic 5)")
    ax9.imshow(pic[8], cmap = "gray")

    plt.show()


