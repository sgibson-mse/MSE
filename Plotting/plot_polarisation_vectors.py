
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
# from Plotting.format import plot_format
#
# cb = plot_format()

class Arrow3D(FancyArrowPatch):
    """
    Class to add an arrow to the graph ie. the field vector.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """
        Overwrites the draw method of FancyArrowPatch object so we can do this in 3D
        :param renderer:
        :return:
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def linearly_polarised_light(omega, t, orientation):

    if orientation=='horizontal':

        x=t
        y = np.sin(2*np.pi*omega*t)
        z = 0.5*np.ones(len(t))

        return x, y, z

    if orientation=='vertical':

        x=t
        y = 0.5*np.ones(len(t))
        z = np.sin(2*np.pi*omega*t)

        return x, y, z

def circularly_polarised_light(omega, t, handedness):

    if handedness=='left':

        x = t
        y = np.cos(2 * np.pi * omega * t)
        z = np.sin(2 * np.pi * omega * t)

        return x, y, z

    if handedness=='right':

        x = t
        y = np.sin(2 * np.pi * omega * t)
        z = np.cos(2 * np.pi * omega * t)

        return x, y, z

    else:
        print('You must specify right or left for handedness!')

    return

def elliptically_polarised(omega, t, theta):

    Ay = np.cos(theta)
    Az = np.sin(theta)
    x = t
    y = Ay*np.cos(2 * np.pi * omega * t)
    z = Az*np.sin(2 * np.pi * omega * t)

    return x, y, z

def plot_animation(t, x, y, z, name):

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")

    ax1.set_title('{} Polarised Light'.format(name))

    #Get rid of all the background grid/fill colour
    ax1.grid(False)
    ax1.xaxis.fill = False
    ax1.yaxis.fill = False
    ax1.zaxis.fill = False
    ax1.set_facecolor('w')

    origin_arrow = np.array([1,0,0])
    origin_circle = np.ones(len(t))

    lines = []

    for i in range(len(t)):
        ax1.view_init(elev=20., azim=-60)
        line1,  = ax1.plot(x[:i], y[:i], z[:i], color='C0')
        line2,  = ax1.plot(origin_circle[:i], y[:i], z[:i], color='black', label='E')
        line3 = Arrow3D([origin_arrow[0],origin_arrow[0]], [origin_arrow[1], y[i]], [origin_arrow[2], z[i]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax1.add_artist(line3)

        lines.append([line1, line2, line3])
        if i==0:
            plt.legend(prop={'size': 10})


    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.tight_layout()
    ani = animation.ArtistAnimation(fig, lines, interval=50, blit=True)

    plt.show()

frequency = 5
phase_shift = 0
theta = 30*np.pi/180.
t = np.arange(0, 1, 0.01)

#x, y, z = linearly_polarised_light(frequency, t, orientation='vertical')

#x, y, z = circularly_polarised_light(frequency, t, handedness='right')

#x, y, z = elliptically_polarised(frequency, t, theta)

plot_animation(t, x, y, z, name='Elliptically')
