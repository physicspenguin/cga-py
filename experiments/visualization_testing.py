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
        pos=point_p_act(plot_points, np.tan(t_slider.value()), [scale, 1]), color=cols
    )


# What happens at update of parameter
def update_subd():
    global plot_points
    global cols
    plot_points, cols = point_cube_gen(
        [0, 0, 0], [2, 2, 2], np.ones(3) * subd_slider.value()
    )
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
t_slider.setLimits([-np.pi, np.pi])
t_slider.setOpts(step=0.005)
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

plot_points, cols = point_cube_gen(
    [0, 0, 0], [2, 2, 2], np.ones(3) * subd_slider.value()
)


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
