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
from time import sleep


########################################
# View generation
########################################
# Generate the App
app = pg.mkQApp("CGA-Py Visualizer")
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
d2 = Dock("Visualization", size=(100, 400))
d3 = Dock("Parameters", size=(50, 400))

# Add the docks to the docking area
area.addDock(d2, "top")
area.addDock(d3, "left")
area.addDock(d1, "bottom")


####################
# Timeline Dock
####################
# make a parameter collection
time_parameters = Parameter.create(name="params", type="group")

# Add a slider
t_slider = time_parameters.addChild(pTypes.SliderParameter(name="Time Slider"))
# change settings
t_slider.setLimits([-np.pi, np.pi])
t_slider.setOpts(step=np.pi / 300)
t_slider.setValue(np.pi / 2 + 0.001)
t_slider.setDefault(np.pi / 2 + 0.001)
# Add a slider
t_value = time_parameters.addChild(
    pTypes.SimpleParameter(type="str", name="Time Input")
)
t_value.setValue(t_slider.value())
t_value.setDefault(t_slider.value())
# What happens at update of parameter
def update_t_slide():
    # t_value.setValue(t_slider.value())
    full_time_update()


def update_t_val():
    t_slider.setValue(eval(t_value.value()))
    full_time_update()


# Create a parameter tree for displaying the slider
time_paramtree = ParameterTree()
time_paramtree.setParameters(time_parameters, showTop=False)

####################
# Parameters dock
####################

general_params = Parameter.create(name="params", type="group")


# Polynomial input
poly_tree = general_params.addChild(Parameter.create(name="Polynomial", type="group"))
poly_box = poly_tree.addChild(pTypes.SimpleParameter(type="str", name="Coefficients"))
poly_box.setValue("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")
poly_box.setDefault("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")


####################
# Trajectory Parameters
####################
traj_tree = general_params.addChild(Parameter.create(name="Trajectories", type="group"))
display_tajectory = traj_tree.addChild(
    pTypes.SimpleParameter(name="Display Trajectories", type="bool")
)
display_tajectory.setValue(False)

p_traj_points = traj_tree.addChild(
    pTypes.SimpleParameter(type="str", name="Trajectory Points")
)
p_traj_points.setValue("[[0,0,0],[1,1,1]]")
p_traj_points.setDefault("[[0,0,0],[1,1,1]]")

p_traj_subds = traj_tree.addChild(
    pTypes.SimpleParameter(type="int", name="Trajectory Subdivisions")
)
p_traj_subds.setValue(200)
p_traj_subds.setDefault(200)

p_traj_width = traj_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Trajectory width")
)
p_traj_width.setValue(2)
p_traj_width.setDefault(2)

####################
# Cube Parameters
####################


cube_tree = general_params.addChild(Parameter.create(name="Cube", type="group"))
display_cube = cube_tree.addChild(
    pTypes.SimpleParameter(name="Display Cube", type="bool")
)
display_cube.setValue(True)
subds = cube_tree.addChild(pTypes.SimpleParameter(name="Subdivisions", type="str"))
subds.setOpts(step=1)
subds.setValue("[10, 10, 10]")
subds.setDefault("[10, 10, 10]")
center = cube_tree.addChild(pTypes.SimpleParameter(name="Center", type="str"))
center.setOpts(step=1)
center.setValue("[0, 0, 0]")
center.setDefault("[0, 0, 0]")
length = cube_tree.addChild(pTypes.SimpleParameter(name="Lengths", type="str"))
length.setOpts(step=1)
length.setValue("[2, 2, 2]")
length.setDefault("[2, 2, 2]")


# Create a parameter tree for displaying the slider
general_paramtree = ParameterTree()
general_paramtree.setParameters(general_params, showTop=False)

a = [0, 0, 0]
poly_coeff = [0.5 * e_123o - e_123i, e_12 - e_3i + 0.5 * e_3o, 1]
cube_points, cols = point_cube_gen(
    eval(center.value()), eval(length.value()), eval(subds.value())
)
traj_points = eval(p_traj_points.value())

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
scatter = gl.GLScatterPlotItem(pos=cube_points, color=cols)


def full_time_update():
    if display_cube.value():
        scatter.setData(
            pos=point_p_act(cube_points, np.tan(t_slider.value()), poly_coeff),
            color=cols,
        )


def update_display_cube():
    if display_cube.value():
        view.addItem(scatter)
    else:
        view.removeItem(scatter)
    full_time_update()


def generatre_trajectory_points(start_point):
    time = np.linspace(0, np.pi, p_traj_subds.value())
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(
                poly_act(np.tan(time[i]), poly_coeff, point(start_point))
            )
        )
    return points


traj_plots = [
    gl.GLLinePlotItem(width=p_traj_width.value()) for i in range(len(traj_points))
]
traj_plots[0].setData(pos=generatre_trajectory_points(traj_points[0]))
traj_plots[1].setData(pos=generatre_trajectory_points(traj_points[1]))


def clear_traj_plots():
    for i in range(len(traj_plots)):
        try:
            view.removeItem(traj_plots[i])
        except ValueError:
            pass


def add_traj_plots():
    for plot in traj_plots:
        view.addItem(plot)


def update_trajectories():
    global traj_points
    global traj_plots
    traj_points = eval(p_traj_points.value())
    clear_traj_plots()
    traj_plots = [gl.GLLinePlotItem(width=5) for i in range(len(traj_points))]
    for i in range(len(traj_plots)):
        traj_plots[i].setData(pos=generatre_trajectory_points(traj_points[i]))
        if display_tajectory.value():
            view.addItem(traj_plots[i])


def update_trajectory_width():
    for i in range(len(traj_plots)):
        traj_plots[i].setData(width=p_traj_width.value())


def update_display_trajectory():
    if display_tajectory.value():
        add_traj_plots()
    else:
        clear_traj_plots()


def update_cube():
    global cube_points
    global cols
    cube_points, cols = point_cube_gen(
        eval(center.value()), eval(length.value()), np.array(eval(subds.value()))
    )
    full_update()


def update_poly_box():
    global poly_coeff
    poly_coeff = eval(poly_box.value())
    full_update()


def full_update():
    update_trajectories()
    full_time_update()


view.addItem(scatter)
# view.addItem(sphere)

# Connect the slider change signals to the update functions
t_slider.sigValueChanged.connect(update_t_slide)
t_value.sigValueChanged.connect(update_t_val)
display_cube.sigValueChanged.connect(update_display_cube)
subds.sigValueChanged.connect(update_cube)
center.sigValueChanged.connect(update_cube)
length.sigValueChanged.connect(update_cube)
poly_box.sigValueChanged.connect(update_poly_box)
display_tajectory.sigValueChanged.connect(update_display_trajectory)
p_traj_points.sigValueChanged.connect(update_trajectories)
p_traj_subds.sigValueChanged.connect(update_trajectories)
p_traj_width.sigValueChanged.connect(update_trajectory_width)

# show the window
win.show()

# The typical stuff for standalone execution
if __name__ == "__main__":
    pg.exec()
