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


########################################
# View generation
########################################
# Generate the App
app = pg.mkQApp("Testing Docking with sliders")
# Make the window in which the app is running
win = QtWidgets.QMainWindow()
# Make the are in which docking is posible and then set it as the central widget
area = DockArea()
win.setCentralWidget(area)

####################
# Create the docks
####################

# Make the two docks. D1 has minimal size and d2 whatever was in the example
d1 = Dock("Timeline", size=(1, 1))
d2 = Dock("Visualization", size=(500, 400))
d3 = Dock("Parameters", size=(50, 400))

# Add the docks to the docking area
area.addDock(d2, "top")
area.addDock(d3, "left")
area.addDock(d1, "bottom")


def full_update():
    scatter.setData(
        pos=point_p_act(plot_points, np.tan(t_slider.value()), poly_coeff), color=cols
    )


####################
# Timeline Dock
####################
# make a parameter collection
time_parameters = Parameter.create(name="params", type="group")

# Add a slider
t_slider = time_parameters.addChild(pTypes.SliderParameter(name="time"))
# change settings
t_slider.setLimits([-np.pi, np.pi])
t_slider.setOpts(step=np.pi / 300)
t_slider.setValue(np.pi / 2)
# What happens at update of parameter
def update_t():
    full_update()


# Create a parameter tree for displaying the slider
time_paramtree = ParameterTree()
time_paramtree.setParameters(time_parameters, showTop=False)
time_paramtree.setWindowTitle("pyqtgraph example: Parameter Tree")

####################
# Parameters dock
####################

general_params = Parameter.create(name="params", type="group")

# Add Spinbox
subd_slider = general_params.addChild(
    pTypes.SimpleParameter(name="subdivisions", type="int")
)
subd_slider.setOpts(step=1)
subd_slider.setValue(10)
# What happens at update of parameter
def update_subd():
    global plot_points
    global cols
    plot_points, cols = point_cube_gen(
        [0, 0, 0], [2, 2, 2], np.ones(3) * subd_slider.value()
    )
    full_update()


# Polynomial input
poly_box = general_params.addChild(
    pTypes.SimpleParameter(type="str", name="Coefficients")
)
poly_box.setValue("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")
poly_box.setDefault("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")


def update_poly_box():
    global poly_coeff
    poly_coeff = eval(poly_box.value())
    full_update()


# Create a parameter tree for displaying the slider
general_paramtree = ParameterTree()
general_paramtree.setParameters(general_params, showTop=False)
general_paramtree.setWindowTitle("pyqtgraph example: Parameter Tree")

a = [0, 0, 0]
scale = transv([0, 0, 0], [1, 2, 3])
poly_coeff = [0.5 * e_123o - e_123i, e_12 - e_3i + 0.5 * e_3o, 1]

plot_points, cols = point_cube_gen(
    [0, 0, 0], [2, 2, 2], np.ones(3) * subd_slider.value()
)

########################################
# Create the view and setup parameters
########################################

# Create a 3d viewport
view = gl.GLViewWidget()
# add axes for viewing pleasure
axes = gl.GLAxisItem()
axes.setSize(3, 3, 3)
view.addItem(axes)

# Add the Tree and Viewer to their respective docks
d1.addWidget(time_paramtree)
d2.addWidget(view)
d3.addWidget(general_paramtree)

# Plot them as scatter
scatter = gl.GLScatterPlotItem(pos=plot_points, color=cols)


# sphere = sphere_gen([1,2,3], 2, rows = 20, cols = 40)


# add them to the viewport
view.addItem(scatter)
# view.addItem(sphere)

# Connect the slider change signals to the update functions
t_slider.sigValueChanged.connect(update_t)
subd_slider.sigValueChanged.connect(update_subd)
poly_box.sigValueChanged.connect(update_poly_box)

# show the window
win.show()

# The typical stuff for standalone execution
if __name__ == "__main__":
    pg.exec()
