import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from cga_py import *
import numpy as np
from numba import jit, njit, prange


# What happens at update of parameter
def update_t():
    scatter.setData(
        pos=point_p_act(plot_points, t_slider.value(), [1, scale]), color=cols
    )


# What happens at update of parameter
def update_subd():
    global plot_points
    global cols
    plot_points, cols = generate_points(-1, 1, subd_slider.value())
    scatter.setData(pos=plot_points, color=cols)


# Generate the App
app = pg.mkQApp("Testing Docking with sliders")
# Make the window in which the app is running
win = QtWidgets.QMainWindow()
# Make the are in which docking is posible and then set it as the central widget
area = DockArea()
win.setCentralWidget(area)

# Make the two docks. D1 has minimal size and d2 whatever was in the example
d1 = Dock("Parameter Tree", size=(1, 1))
d2 = Dock("Visualization", size=(500, 400))

# Add the docks to the docking area
area.addDock(d1, "bottom")
area.addDock(d2, "top")

# make a parameter collection
p = Parameter.create(name="params", type="group")

# Add a slider
t_slider = p.addChild(pTypes.SliderParameter(name="t"))
# Extract slider for convenience
# slider = p.child('t')
# change settings
t_slider.setLimits([-10, 20])
t_slider.setOpts(step=0.01)
t_slider.setValue(0)

# Add Spinbox
subd_slider = p.addChild(pTypes.SliderParameter(name="subdivisions"))

subd_slider.setLimits([2, 20])
subd_slider.setOpts(step=1)
subd_slider.setValue(10)


# Create a parameter tree for displaying the slider
t = ParameterTree()
t.setParameters(p, showTop=False)
t.setWindowTitle("pyqtgraph example: Parameter Tree")

a = [0, 0, 0]
scale = transv([0, 0, 0], [1, 2, 3])


@njit(parallel=True, cache=True)
def generate_points(start, end, subd):
    points = np.linspace(start, end, subd)
    plot_points = np.zeros((subd**3, 3))
    colors = np.ones((subd**3, 4))
    for x in range(subd):
        for y in range(subd):
            for z in range(subd):
                plot_points[subd * subd * x + subd * y + z, 0] = points[x]
                plot_points[subd * subd * x + subd * y + z, 1] = points[y]
                plot_points[subd * subd * x + subd * y + z, 2] = points[z]
                colors[subd * subd * x + subd * y + z, 0] = points[x]
                colors[subd * subd * x + subd * y + z, 1] = points[y]
                colors[subd * subd * x + subd * y + z, 2] = points[z]
    return plot_points, ((colors - start) / (end - start))


plot_points, cols = generate_points(-1, 1, subd_slider.value())


# Create a 3d viewport
view = gl.GLViewWidget()
# add axes for viewing pleasure
axes = gl.GLAxisItem()
axes.setSize(3, 3, 3)
view.addItem(axes)

# Add the Tree and Viewer to their respective docks
d1.addWidget(t)
d2.addWidget(view)

# Plot them as scatter
scatter = gl.GLScatterPlotItem(pos=plot_points, color=cols)

# add them to the viewport
view.addItem(scatter)

# Connect the slider change signals to the update functions
t_slider.sigValueChanged.connect(update_t)
subd_slider.sigValueChanged.connect(update_subd)

# show the window
win.show()

# The typical stuff for standalone execution
if __name__ == "__main__":
    pg.exec()
