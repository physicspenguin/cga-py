#!/usr/bin/env python

from cga_py import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import numpy as np

pg.setConfigOption("useOpenGL", True)


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
p_t_slider = time_parameters.addChild(pTypes.SliderParameter(name="Time Slider"))
# change settings
p_t_slider.setLimits([-np.pi / 2, np.pi / 2])
p_t_slider.setOpts(step=np.pi / 300)
p_t_slider.setValue(np.pi / 2 + 0.001)
p_t_slider.setDefault(np.pi / 2 + 0.001)
# Add a Value Input for Slider
p_t_value = time_parameters.addChild(
    pTypes.SimpleParameter(type="str", name="Time Input")
)
p_t_value.setValue(p_t_slider.value())
p_t_value.setDefault(p_t_slider.value())


# What happens at update of parameter
def update_t_slide():
    full_time_update()


def update_t_val():
    p_t_slider.setValue(eval(p_t_value.value()))
    full_time_update()


# Create a parameter tree for displaying the slider
time_paramtree = ParameterTree()
time_paramtree.setParameters(time_parameters, showTop=False)

####################
# Parameters dock
####################

general_params = Parameter.create(name="params", type="group")

p_update_all = general_params.addChild(pTypes.ActionParameter(name="Update All"))

p_execute_code = general_params.addChild(pTypes.ActionParameter(name="Execute Code"))

p_shell_interface = general_params.addChild(
    pTypes.TextParameter(name="Custom Code", expanded=False)
)
p_shell_interface.setValue(
    "#This Field can execute custom code. Variables must be declared with the global keyword!"
)

# View Tree
view_tree = general_params.addChild(
    Parameter.create(name="View Box Settings", type="group", expanded=False)
)

p_background_color = view_tree.addChild(pTypes.ColorParameter(name="Backgroung Color"))
p_background_color.setValue(pg.mkColor(0.0))
p_background_color.setDefault(pg.mkColor(0.0))


p_show_axes = view_tree.addChild(pTypes.SimpleParameter(name="Show Axes", type="bool"))
p_show_axes.setValue(True)
p_show_axes.setDefault(True)

p_axes_size = view_tree.addChild(
    pTypes.SimpleParameter(name="Axes Lengths", type="str")
)
p_axes_size.setValue("[3,3,3]")
p_axes_size.setDefault("[3,3,3]")

p_blend = view_tree.addChild(pTypes.ListParameter(name="Blend Method"))
p_blend.setLimits(["additive", "translucent", "opaque"])
p_blend.setValue("additive")
p_blend.setDefault("additive")


####################
# Polynomial coefficients
####################
poly_tree = general_params.addChild(
    Parameter.create(name="Polynomial", type="group", expanded=False)
)

p_update_coeff = poly_tree.addChild(pTypes.ActionParameter(name="Update Coefficients"))


p_use_main_coeff = poly_tree.addChild(
    pTypes.SimpleParameter(name="Use Main Polynomial", type="bool")
)
p_use_main_coeff.setValue(True)


p_main_poly_coeff = poly_tree.addChild(
    pTypes.TextParameter(name="Main Coeff", expanded=False)
)
p_main_poly_coeff.setValue("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")
p_main_poly_coeff.setDefault("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")

x = 0
y = 0
z = 1

p_first_poly_coeff = poly_tree.addChild(
    pTypes.TextParameter(name="First Coeff", expanded=False)
)
p_first_poly_coeff.setValue(
    "[-1/2*(2*x**4*e_3i-x**4*e_3o+4*x**2*y**2*e_3i-2*x**2*y**2*e_3o+4*x**2*z**2*e_3i-2*x**2*z**2*e_3o+2*y**4*e_3i-y**4*e_3o+4*y**2*z**2*e_3i-2*y**2*z**2*e_3o+2*z**4*e_3i-z**4*e_3o-8*x**3*e_13+8*x**3*e_2i-4*x**3*e_2o-8*x**2*y*e_1i+4*x**2*y*e_1o-8*x**2*y*e_23-8*x*y**2*e_13+8*x*y**2*e_2i-4*x*y**2*e_2o-8*x*z**2*e_13+8*x*z**2*e_2i-4*x*z**2*e_2o-8*y**3*e_1i+4*y**3*e_1o-8*y**3*e_23-8*y*z**2*e_1i+4*y*z**2*e_1o-8*y*z**2*e_23-32*x**2*e_12-16*x**2*e_3i+8*x**2*e_3o+32*x*z*e_1i-16*x*z*e_1o+32*x*z*e_23-32*y**2*e_12-16*y**2*e_3i+8*y**2*e_3o-32*y*z*e_13+32*y*z*e_2i-16*y*z*e_2o+16*z**2*e_3i-8*z**2*e_3o+32*x*e_13-32*x*e_2i+16*x*e_2o+32*y*e_1i-16*y*e_1o+32*y*e_23+32*e_3i-16*e_3o)/(x**2+y**2+z**2+4)**2, 1]"
)
p_first_poly_coeff.setDefault(
    "[-1/2*(2*x**4*e_3i-x**4*e_3o+4*x**2*y**2*e_3i-2*x**2*y**2*e_3o+4*x**2*z**2*e_3i-2*x**2*z**2*e_3o+2*y**4*e_3i-y**4*e_3o+4*y**2*z**2*e_3i-2*y**2*z**2*e_3o+2*z**4*e_3i-z**4*e_3o-8*x**3*e_13+8*x**3*e_2i-4*x**3*e_2o-8*x**2*y*e_1i+4*x**2*y*e_1o-8*x**2*y*e_23-8*x*y**2*e_13+8*x*y**2*e_2i-4*x*y**2*e_2o-8*x*z**2*e_13+8*x*z**2*e_2i-4*x*z**2*e_2o-8*y**3*e_1i+4*y**3*e_1o-8*y**3*e_23-8*y*z**2*e_1i+4*y*z**2*e_1o-8*y*z**2*e_23-32*x**2*e_12-16*x**2*e_3i+8*x**2*e_3o+32*x*z*e_1i-16*x*z*e_1o+32*x*z*e_23-32*y**2*e_12-16*y**2*e_3i+8*y**2*e_3o-32*y*z*e_13+32*y*z*e_2i-16*y*z*e_2o+16*z**2*e_3i-8*z**2*e_3o+32*x*e_13-32*x*e_2i+16*x*e_2o+32*y*e_1i-16*y*e_1o+32*y*e_23+32*e_3i-16*e_3o)/(x**2+y**2+z**2+4)**2, 1]"
)

p_second_poly_coeff = poly_tree.addChild(
    pTypes.TextParameter(name="Second Coeff", expanded=False)
)
p_second_poly_coeff.setValue(
    "[-16/(x**2+y**2+z**2+4)**2*e_12*x**2-16/(x**2+y**2+z**2+4)**2*e_12*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x**3+16/(x**2+y**2+z**2+4)**2*e_13*x-4/(x**2+y**2+z**2+4)**2*e_1i*y**3+16/(x**2+y**2+z**2+4)**2*e_1i*y+2/(x**2+y**2+z**2+4)**2*e_1o*y**3-8/(x**2+y**2+z**2+4)**2*e_1o*y-4/(x**2+y**2+z**2+4)**2*e_23*y**3+16/(x**2+y**2+z**2+4)**2*e_23*y+4/(x**2+y**2+z**2+4)**2*e_2i*x**3-16/(x**2+y**2+z**2+4)**2*e_2i*x-2/(x**2+y**2+z**2+4)**2*e_2o*x**3+8/(x**2+y**2+z**2+4)**2*e_2o*x+16/(x**2+y**2+z**2+4)**2*e_3i+1/(x**2+y**2+z**2+4)**2*e_3i*x**4+1/(x**2+y**2+z**2+4)**2*e_3i*y**4+1/(x**2+y**2+z**2+4)**2*e_3i*z**4-8/(x**2+y**2+z**2+4)**2*e_3i*x**2-8/(x**2+y**2+z**2+4)**2*e_3i*y**2+8/(x**2+y**2+z**2+4)**2*e_3i*z**2-8/(x**2+y**2+z**2+4)**2*e_3o-1/2/(x**2+y**2+z**2+4)**2*e_3o*x**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*y**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*z**4+4/(x**2+y**2+z**2+4)**2*e_3o*x**2+4/(x**2+y**2+z**2+4)**2*e_3o*y**2-4/(x**2+y**2+z**2+4)**2*e_3o*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*y**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*z**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*z**2-1/(x**2+y**2+z**2+4)**2*e_3o*z**2*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*z**2-16/(x**2+y**2+z**2+4)**2*e_13*y*z-4/(x**2+y**2+z**2+4)**2*e_1i*x**2*y-4/(x**2+y**2+z**2+4)**2*e_1i*y*z**2+16/(x**2+y**2+z**2+4)**2*e_1i*x*z+2/(x**2+y**2+z**2+4)**2*e_1o*x**2*y+2/(x**2+y**2+z**2+4)**2*e_1o*y*z**2-8/(x**2+y**2+z**2+4)**2*e_1o*x*z-4/(x**2+y**2+z**2+4)**2*e_23*x**2*y-4/(x**2+y**2+z**2+4)**2*e_23*y*z**2+16/(x**2+y**2+z**2+4)**2*e_23*x*z+4/(x**2+y**2+z**2+4)**2*e_2i*x*y**2+4/(x**2+y**2+z**2+4)**2*e_2i*x*z**2+16/(x**2+y**2+z**2+4)**2*e_2i*y*z-2/(x**2+y**2+z**2+4)**2*e_2o*x*y**2-2/(x**2+y**2+z**2+4)**2*e_2o*x*z**2-8/(x**2+y**2+z**2+4)**2*e_2o*y*z+e_12-e_3i+1/2*e_3o, 1]"
)
p_second_poly_coeff.setDefault(
    "[-16/(x**2+y**2+z**2+4)**2*e_12*x**2-16/(x**2+y**2+z**2+4)**2*e_12*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x**3+16/(x**2+y**2+z**2+4)**2*e_13*x-4/(x**2+y**2+z**2+4)**2*e_1i*y**3+16/(x**2+y**2+z**2+4)**2*e_1i*y+2/(x**2+y**2+z**2+4)**2*e_1o*y**3-8/(x**2+y**2+z**2+4)**2*e_1o*y-4/(x**2+y**2+z**2+4)**2*e_23*y**3+16/(x**2+y**2+z**2+4)**2*e_23*y+4/(x**2+y**2+z**2+4)**2*e_2i*x**3-16/(x**2+y**2+z**2+4)**2*e_2i*x-2/(x**2+y**2+z**2+4)**2*e_2o*x**3+8/(x**2+y**2+z**2+4)**2*e_2o*x+16/(x**2+y**2+z**2+4)**2*e_3i+1/(x**2+y**2+z**2+4)**2*e_3i*x**4+1/(x**2+y**2+z**2+4)**2*e_3i*y**4+1/(x**2+y**2+z**2+4)**2*e_3i*z**4-8/(x**2+y**2+z**2+4)**2*e_3i*x**2-8/(x**2+y**2+z**2+4)**2*e_3i*y**2+8/(x**2+y**2+z**2+4)**2*e_3i*z**2-8/(x**2+y**2+z**2+4)**2*e_3o-1/2/(x**2+y**2+z**2+4)**2*e_3o*x**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*y**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*z**4+4/(x**2+y**2+z**2+4)**2*e_3o*x**2+4/(x**2+y**2+z**2+4)**2*e_3o*y**2-4/(x**2+y**2+z**2+4)**2*e_3o*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*y**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*z**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*z**2-1/(x**2+y**2+z**2+4)**2*e_3o*z**2*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*z**2-16/(x**2+y**2+z**2+4)**2*e_13*y*z-4/(x**2+y**2+z**2+4)**2*e_1i*x**2*y-4/(x**2+y**2+z**2+4)**2*e_1i*y*z**2+16/(x**2+y**2+z**2+4)**2*e_1i*x*z+2/(x**2+y**2+z**2+4)**2*e_1o*x**2*y+2/(x**2+y**2+z**2+4)**2*e_1o*y*z**2-8/(x**2+y**2+z**2+4)**2*e_1o*x*z-4/(x**2+y**2+z**2+4)**2*e_23*x**2*y-4/(x**2+y**2+z**2+4)**2*e_23*y*z**2+16/(x**2+y**2+z**2+4)**2*e_23*x*z+4/(x**2+y**2+z**2+4)**2*e_2i*x*y**2+4/(x**2+y**2+z**2+4)**2*e_2i*x*z**2+16/(x**2+y**2+z**2+4)**2*e_2i*y*z-2/(x**2+y**2+z**2+4)**2*e_2o*x*y**2-2/(x**2+y**2+z**2+4)**2*e_2o*x*z**2-8/(x**2+y**2+z**2+4)**2*e_2o*y*z+e_12-e_3i+1/2*e_3o, 1]"
)


####################
# Trajectory Parameters
####################
traj_tree = general_params.addChild(
    Parameter.create(name="Trajectories", type="group", expanded=False)
)


##########
# Main Trajectory
##########

traj_main_tree = traj_tree.addChild(
    Parameter.create(name="Main Trajectories", type="group", expanded=False)
)

p_traj_main_points_update = traj_main_tree.addChild(
    pTypes.ActionParameter(name="Update Trajectories")
)

p_display_traj_main = traj_main_tree.addChild(
    pTypes.SimpleParameter(name="Display Trajectories", type="bool")
)
p_display_traj_main.setValue(False)

p_display_traj_points_main = traj_main_tree.addChild(
    pTypes.SimpleParameter(name="Display Points", type="bool")
)
p_display_traj_points_main.setValue(False)
p_traj_main_point_width = traj_main_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Point Size")
)
p_traj_main_point_width.setValue(10)
p_traj_main_point_width.setDefault(10)

p_traj_main_points = traj_main_tree.addChild(
    pTypes.TextParameter(name="Trajectory Points", expanded=False)
)
p_traj_main_points.setValue(
    "[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20, endpoint=False)]"
)
p_traj_main_points.setDefault(
    "[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20, endpoint=False)]"
)

p_traj_main_subds = traj_main_tree.addChild(
    pTypes.SimpleParameter(type="int", name="Trajectory Subdivisions")
)
p_traj_main_subds.setValue(50)
p_traj_main_subds.setDefault(50)

p_traj_main_width = traj_main_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Trajectory width")
)
p_traj_main_width.setValue(1)
p_traj_main_width.setDefault(1)
p_traj_main_c_map = traj_main_tree.addChild(
    pTypes.ColorMapParameter(name="Trajectory Colors")
)

##########
# First Trajectory
##########

traj_first_tree = traj_tree.addChild(
    Parameter.create(name="First Trajectories", type="group", expanded=False)
)

p_traj_first_points_update = traj_first_tree.addChild(
    pTypes.ActionParameter(name="Update Trajectories")
)

p_display_traj_first = traj_first_tree.addChild(
    pTypes.SimpleParameter(name="Display Trajectories", type="bool")
)
p_display_traj_first.setValue(False)

p_display_traj_points_first = traj_first_tree.addChild(
    pTypes.SimpleParameter(name="Display Points", type="bool")
)
p_display_traj_points_first.setValue(False)
p_display_traj_points_first.setValue(False)
p_traj_first_point_width = traj_first_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Point Size")
)
p_traj_first_point_width.setValue(10)
p_traj_first_point_width.setDefault(10)

p_traj_first_points = traj_first_tree.addChild(
    pTypes.TextParameter(name="Trajectory Points", expanded=False)
)
p_traj_first_points.setValue(
    "[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20, endpoint=False)]"
)
p_traj_first_points.setDefault(
    "[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20, endpoint=False)]"
)

p_traj_first_subds = traj_first_tree.addChild(
    pTypes.SimpleParameter(type="int", name="Trajectory Subdivisions")
)
p_traj_first_subds.setValue(50)
p_traj_first_subds.setDefault(50)

p_traj_first_width = traj_first_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Trajectory width")
)
p_traj_first_width.setValue(1)
p_traj_first_width.setDefault(1)
p_traj_first_c_map = traj_first_tree.addChild(
    pTypes.ColorMapParameter(name="Trajectory Colors")
)

##########
# Second Trajectory
##########

traj_second_tree = traj_tree.addChild(
    Parameter.create(name="Second Trajectories", type="group", expanded=False)
)

p_traj_second_points_update = traj_second_tree.addChild(
    pTypes.ActionParameter(name="Update Trajectories")
)

p_display_traj_second = traj_second_tree.addChild(
    pTypes.SimpleParameter(name="Display Trajectories", type="bool")
)
p_display_traj_second.setValue(False)

p_display_traj_points_second = traj_second_tree.addChild(
    pTypes.SimpleParameter(name="Display Points", type="bool")
)
p_display_traj_points_second.setValue(False)
p_display_traj_points_second.setValue(False)
p_traj_second_point_width = traj_second_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Point Size")
)
p_traj_second_point_width.setValue(10)
p_traj_second_point_width.setDefault(10)

p_traj_second_points = traj_second_tree.addChild(
    pTypes.TextParameter(name="Trajectory Points", expanded=False)
)
p_traj_second_points.setValue(
    "np.array([[np.sin(t),0,np.cos(t)] for t in np.linspace(0,2*np.pi,20,endpoint=False)])*1.5+np.array([2.5,0,0])"
)
p_traj_second_points.setDefault(
    "np.array([[np.sin(t),0,np.cos(t)] for t in np.linspace(0,2*np.pi,20,endpoint=False)])*1.5+np.array([2.5,0,0])"
)

p_traj_second_subds = traj_second_tree.addChild(
    pTypes.SimpleParameter(type="int", name="Trajectory Subdivisions")
)
p_traj_second_subds.setValue(50)
p_traj_second_subds.setDefault(50)

p_traj_second_width = traj_second_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Trajectory width")
)
p_traj_second_width.setValue(1)
p_traj_second_width.setDefault(1)
p_traj_second_c_map = traj_second_tree.addChild(
    pTypes.ColorMapParameter(name="Trajectory Colors")
)


####################
# Cube Parameters
####################
cube_tree = general_params.addChild(
    Parameter.create(name="Cube", type="group", expanded=False)
)
p_display_cube = cube_tree.addChild(
    pTypes.SimpleParameter(name="Display Cube", type="bool")
)
p_display_cube.setValue(True)
p_cube_subds = cube_tree.addChild(
    pTypes.SimpleParameter(name="Subdivisions", type="str")
)
p_cube_subds.setOpts(step=1)
p_cube_subds.setValue("[10, 10, 10]")
p_cube_subds.setDefault("[10, 10, 10]")
p_cube_center = cube_tree.addChild(pTypes.SimpleParameter(name="Center", type="str"))
p_cube_center.setOpts(step=1)
p_cube_center.setValue("[0, 0, 0]")
p_cube_center.setDefault("[0, 0, 0]")
p_cube_length = cube_tree.addChild(pTypes.SimpleParameter(name="Lengths", type="str"))
p_cube_length.setOpts(step=1)
p_cube_length.setValue("[2, 2, 2]")
p_cube_length.setDefault("[2, 2, 2]")


####################
# Point Parameters
####################
points_tree = general_params.addChild(
    Parameter.create(name="Points", type="group", expanded=False)
)
p_update_points = points_tree.addChild(pTypes.ActionParameter(name="Update Points"))
p_display_points = points_tree.addChild(
    pTypes.SimpleParameter(name="Display Points", type="bool")
)
p_display_points.setValue(False)
p_points_width = points_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Point Size")
)
p_points_width.setValue(10)
p_points_width.setDefault(10)
p_points_params = points_tree.addChild(
    pTypes.TextParameter(name="Point Coordinates", expanded=False)
)
p_points_params.setValue("[[1, 0, 0],[0, -1, 0],[-1, 0, 0],[0, 1, 0],[0,0,1],[0,0,-1]]")
p_points_params.setDefault(
    "[[1, 0, 0],[0, -1, 0],[-1, 0, 0],[0, 1, 0],[0,0,1],[0,0,-1]]"
)
p_points_c_map = points_tree.addChild(pTypes.ColorMapParameter(name="Point Colors"))


####################
# Sphere Parameters
####################
sphere_tree = general_params.addChild(
    Parameter.create(name="Spheres", type="group", expanded=False)
)
p_update_spheres = sphere_tree.addChild(pTypes.ActionParameter(name="Update Spheres"))
p_display_sphere = sphere_tree.addChild(
    pTypes.SimpleParameter(name="Display Spheres", type="bool")
)
p_display_sphere.setValue(False)
p_sphere_params = sphere_tree.addChild(
    pTypes.TextParameter(name="Sphere Parameters", expanded=False)
)
p_sphere_params.setValue(
    "[[[1, 0, 0],0.5],[[0, -1, 0],0.5],[[-1, 0, 0],0.5],[[0, 1, 0],0.5],[[0,0,1],0.5],[[0,0,-1],0.5]]"
)
p_sphere_params.setDefault(
    "[[[1, 0, 0],0.5],[[0, -1, 0],0.5],[[-1, 0, 0],0.5],[[0, 1, 0],0.5],[[0,0,1],0.5],[[0,0,-1],0.5]]"
)
p_sphere_c_map = sphere_tree.addChild(pTypes.ColorMapParameter(name="Sphere Colors"))


####################
# Generated Cyclic Parameters
####################

cyclic_tree = general_params.addChild(
    Parameter.create(name="Generated Cyclic", type="group", expanded=False)
)
p_update_cyclics = cyclic_tree.addChild(pTypes.ActionParameter(name="Update Cyclic"))
p_display_cyclic = cyclic_tree.addChild(
    pTypes.SimpleParameter(name="Display Generated Cyclic", type="bool")
)
p_display_cyclic.setValue(False)
p_cyclic_params = cyclic_tree.addChild(
    pTypes.SimpleParameter(name="Generating Point", type="str")
)
p_cyclic_params.setValue("[1,0,0]")
p_cyclic_params.setDefault("[1,0,0]")
p_cyclic_mesh = cyclic_tree.addChild(
    pTypes.SimpleParameter(name="Surface Subdivisions", type="int")
)
p_cyclic_mesh.setValue(20)
p_cyclic_mesh.setDefault(20)
p_cyclic_subd = cyclic_tree.addChild(
    pTypes.SimpleParameter(name="Line Subdivisions", type="int")
)
p_cyclic_subd.setValue(50)
p_cyclic_subd.setDefault(50)
p_cyclic_width = cyclic_tree.addChild(
    pTypes.SimpleParameter(name="Linewidth", type="int")
)
p_cyclic_width.setValue(1)
p_cyclic_width.setDefault(1)
p_cyclic_c_map = cyclic_tree.addChild(pTypes.ColorMapParameter(name="Factor Colors"))


####################
# Parameter Initialization
####################

general_paramtree = ParameterTree()
general_paramtree.setParameters(general_params, showTop=False)

blend = p_blend.value()

main_poly_coeff = eval(p_main_poly_coeff.value())
first_poly_coeff = eval(p_first_poly_coeff.value())
second_poly_coeff = eval(p_second_poly_coeff.value())
cube_points, cols = point_cube_gen(
    eval(p_cube_center.value()), eval(p_cube_length.value()), eval(p_cube_subds.value())
)


traj_main_points = eval(p_traj_main_points.value())
traj_main_points_updated = eval(p_traj_main_points.value())
traj_main_point_colors = p_traj_main_c_map.value().getLookupTable(
    nPts=len(traj_main_points), mode=pg.ColorMap.FLOAT
)
traj_main_scatter = gl.GLScatterPlotItem(
    pos=traj_main_points_updated,
    color=traj_main_point_colors,
    size=p_traj_main_point_width.value(),
)
traj_main_scatter.setGLOptions(blend)
traj_main_plots = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
traj_main_colors = p_traj_main_c_map.value().getLookupTable(
    nPts=len(traj_main_plots), mode=pg.ColorMap.QCOLOR
)
traj_first_points = eval(p_traj_first_points.value())
traj_first_points_updated = eval(p_traj_first_points.value())
traj_first_point_colors = p_traj_first_c_map.value().getLookupTable(
    nPts=len(traj_first_points), mode=pg.ColorMap.FLOAT
)
traj_first_scatter = gl.GLScatterPlotItem(
    pos=traj_first_points_updated,
    color=traj_first_point_colors,
    size=p_traj_first_point_width.value(),
)
traj_first_scatter.setGLOptions(blend)
traj_first_plots = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
traj_first_colors = p_traj_first_c_map.value().getLookupTable(
    nPts=len(traj_first_plots), mode=pg.ColorMap.QCOLOR
)

traj_second_points = eval(p_traj_second_points.value())
traj_second_points_updated = eval(p_traj_second_points.value())
traj_second_point_colors = p_traj_second_c_map.value().getLookupTable(
    nPts=len(traj_second_points), mode=pg.ColorMap.FLOAT
)
traj_second_scatter = gl.GLScatterPlotItem(
    pos=traj_second_points_updated,
    color=traj_second_point_colors,
    size=p_traj_second_point_width.value(),
)
traj_second_scatter.setGLOptions(blend)
traj_second_plots = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
traj_second_colors = p_traj_second_c_map.value().getLookupTable(
    nPts=len(traj_second_plots), mode=pg.ColorMap.QCOLOR
)


points_coordinates = eval(p_points_params.value())
points_current_coordinates = eval(p_points_params.value())
points_colors = p_points_c_map.value().getLookupTable(
    nPts=len(points_coordinates), mode=pg.ColorMap.FLOAT
)
points_scatter = gl.GLScatterPlotItem(pos=points_coordinates, color=points_colors)
points_scatter.setGLOptions("translucent")

unit_sphere = gl.MeshData.sphere(20, 20)
sphere_params = eval(p_sphere_params.value())
sphere_centers = [sphere_params[i][0] for i in range(len(sphere_params))]
sphere_radii = [sphere_params[i][1] for i in range(len(sphere_params))]
view_spheres = [gl.GLMeshItem(meshdata=unit_sphere, color=(0.5, 0.5, 0.5, 1))]
sphere_colors = p_sphere_c_map.value().getLookupTable(
    nPts=len(view_spheres), mode=pg.ColorMap.QCOLOR
)


cyclic_gen_point = eval(p_cyclic_params.value())
cyclic_colors = p_cyclic_c_map.value().getLookupTable(nPts=2, mode=pg.ColorMap.QCOLOR)
cyclic_points = [eval(p_cyclic_params.value()) for i in range(p_cyclic_mesh.value())]
cyclic_first_plots = [gl.GLLinePlotItem()]
cyclic_second_plots = [gl.GLLinePlotItem()]


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

# Plot them as cube_plot
cube_plot = gl.GLScatterPlotItem(pos=cube_points, color=cols)
cube_plot.setGLOptions(blend)


def update_bg_color():
    view.setBackgroundColor(p_background_color.value())


def update_show_axes():
    if p_show_axes.value():
        axes.setVisible(True)
    else:
        axes.setVisible(False)


def update_axes_size():
    scale = eval(p_axes_size.value())
    axes.setSize(scale[0], scale[1], scale[2])


####################
# Acting
####################


def point_p_act_main_on_points(point_arr):
    return point_p_act(np.tan(p_t_slider.value()), main_poly_coeff, point_arr)


def point_p_act_factorization_on_points(point_arr):
    return point_p_act(
        np.tan(p_t_slider.value()),
        second_poly_coeff,
        point_p_act(np.tan(p_t_slider.value()), first_poly_coeff, point_arr),
    )


def act_main_on_points(param, actpoint):
    return poly_act(np.tan(param), main_poly_coeff, actpoint)


def act_first_on_points(param, actpoint):
    return poly_act(np.tan(param), first_poly_coeff, actpoint)


def act_second_on_points(param, actpoint):
    return poly_act(np.tan(param), second_poly_coeff, actpoint)


def act_factorization_on_points(param, actpoint):
    return poly_act(
        np.tan(param),
        second_poly_coeff,
        poly_act(np.tan(param), first_poly_coeff, actpoint),
    )


########################################
# Trajectories
########################################

####################
# main
####################


def generate_traj_main_points(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_traj_main_subds.value())
    points = np.empty((len(time), 3))
    if p_use_main_coeff.value():
        for i in range(len(time)):
            points[i] = np.real(
                point_to_cartesian(act_main_on_points(time[i], point(start_point)))
            )
    else:
        for i in range(len(time)):
            points[i] = np.real(
                point_to_cartesian(
                    act_factorization_on_points(time[i], point(start_point))
                )
            )
    return points


def clear_traj_main_plots():
    for i in range(len(traj_main_plots)):
        try:
            view.removeItem(traj_main_plots[i])
        except ValueError:
            pass


def add_traj_main_plots():
    for plot in traj_main_plots:
        view.addItem(plot)
        plot.setGLOptions(blend)


def update_traj_main():
    global traj_main_points
    global traj_main_plots
    global traj_main_points_updated
    traj_main_points = eval(p_traj_main_points.value())
    traj_main_points_updated = eval(p_traj_main_points.value())
    clear_traj_main_plots()
    traj_main_plots = [gl.GLLinePlotItem() for i in range(len(traj_main_points))]
    for i in range(len(traj_main_plots)):
        traj_main_plots[i].setData(pos=generate_traj_main_points(traj_main_points[i]))
    update_display_traj_main()
    update_display_traj_points_main()


def update_traj_main_width():
    for i in range(len(traj_main_plots)):
        traj_main_plots[i].setData(width=p_traj_main_width.value())


def update_traj_main_colors():
    global traj_main_colors
    traj_main_colors = p_traj_main_c_map.value().getLookupTable(
        nPts=len(traj_main_plots), mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(traj_main_plots)):
        traj_main_plots[i].setData(color=traj_main_colors[i])


def update_display_traj_main():
    if p_display_traj_main.value():
        add_traj_main_plots()
        update_traj_main_width()
        update_traj_main_colors()
    else:
        clear_traj_main_plots()


def update_traj_main_point_width():
    traj_main_scatter.setData(
        pos=traj_main_points_updated, size=p_traj_main_point_width.value()
    )


def update_traj_main_point_colors():
    global traj_main_point_colors
    traj_main_point_colors = p_traj_main_c_map.value().getLookupTable(
        nPts=len(traj_main_points), mode=pg.ColorMap.FLOAT
    )
    traj_main_scatter.setData(
        pos=traj_main_points_updated,
        color=traj_main_point_colors,
        size=p_traj_main_point_width.value(),
    )


def update_time_traj_main_points():
    if p_display_traj_points_main.value():
        for i in range(len(traj_main_points_updated)):
            traj_main_points_updated[i] = point_to_cartesian(
                act_main_on_points(p_t_slider.value(), point(traj_main_points[i]))
            )
        traj_main_scatter.setData(
            pos=traj_main_points_updated,
            color=traj_main_point_colors,
            size=p_traj_main_point_width.value(),
        )


def update_display_traj_points_main():
    if p_display_traj_points_main.value():
        update_traj_main_point_width()
        update_traj_main_point_colors()
        update_time_traj_main_points()
        traj_main_scatter.setVisible(True)
    else:
        traj_main_scatter.setVisible(False)


####################
# first
####################


def generate_traj_first_points(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_traj_first_subds.value())
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(act_first_on_points(time[i], point(start_point)))
        )
    return points


def clear_traj_first_plots():
    for i in range(len(traj_first_plots)):
        try:
            view.removeItem(traj_first_plots[i])
        except ValueError:
            pass


def add_traj_first_plots():
    for plot in traj_first_plots:
        view.addItem(plot)
        plot.setGLOptions(blend)


def update_traj_first():
    global traj_first_points
    global traj_first_plots
    global traj_first_points_updated
    traj_first_points = eval(p_traj_first_points.value())
    traj_first_points_updated = eval(p_traj_first_points.value())
    clear_traj_first_plots()
    traj_first_plots = [gl.GLLinePlotItem() for i in range(len(traj_first_points))]
    for i in range(len(traj_first_plots)):
        traj_first_plots[i].setData(
            pos=generate_traj_first_points(traj_first_points[i])
        )
    update_display_traj_first()
    update_display_traj_points_first()


def update_traj_first_width():
    for i in range(len(traj_first_plots)):
        traj_first_plots[i].setData(width=p_traj_first_width.value())


def update_traj_first_colors():
    global traj_first_colors
    traj_first_colors = p_traj_first_c_map.value().getLookupTable(
        nPts=len(traj_first_plots), mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(traj_first_plots)):
        traj_first_plots[i].setData(color=traj_first_colors[i])


def update_display_traj_first():
    if p_display_traj_first.value():
        add_traj_first_plots()
        update_traj_first_width()
        update_traj_first_colors()
    else:
        clear_traj_first_plots()


def update_traj_first_point_width():
    traj_first_scatter.setData(
        pos=traj_first_points_updated, size=p_traj_first_point_width.value()
    )


def update_traj_first_point_colors():
    global traj_first_point_colors
    traj_first_point_colors = p_traj_first_c_map.value().getLookupTable(
        nPts=len(traj_first_points), mode=pg.ColorMap.FLOAT
    )
    traj_first_scatter.setData(
        pos=traj_first_points_updated,
        color=traj_first_point_colors,
        size=p_traj_first_point_width.value(),
    )


def update_time_traj_first_points():
    if p_display_traj_points_first.value():
        for i in range(len(traj_first_points_updated)):
            traj_first_points_updated[i] = point_to_cartesian(
                act_first_on_points(p_t_slider.value(), point(traj_first_points[i]))
            )
        traj_first_scatter.setData(
            pos=traj_first_points_updated,
            color=traj_first_point_colors,
            size=p_traj_first_point_width.value(),
        )


def update_display_traj_points_first():
    if p_display_traj_points_first.value():
        update_traj_first_point_width()
        update_traj_first_point_colors()
        update_time_traj_first_points()
        traj_first_scatter.setVisible(True)
    else:
        traj_first_scatter.setVisible(False)


####################
# second
####################


def generate_traj_second_points(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_traj_second_subds.value())
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(act_second_on_points(time[i], point(start_point)))
        )
    return points


def clear_traj_second_plots():
    for i in range(len(traj_second_plots)):
        try:
            view.removeItem(traj_second_plots[i])
        except ValueError:
            pass


def add_traj_second_plots():
    for plot in traj_second_plots:
        view.addItem(plot)
        plot.setGLOptions(blend)


def update_traj_second():
    global traj_second_points
    global traj_second_plots
    global traj_second_points_updated
    traj_second_points = eval(p_traj_second_points.value())
    traj_second_points_updated = eval(p_traj_second_points.value())
    clear_traj_second_plots()
    traj_second_plots = [gl.GLLinePlotItem() for i in range(len(traj_second_points))]
    for i in range(len(traj_second_plots)):
        traj_second_plots[i].setData(
            pos=generate_traj_second_points(traj_second_points[i])
        )
    update_display_traj_second()
    update_display_traj_points_second()


def update_traj_second_width():
    for i in range(len(traj_second_plots)):
        traj_second_plots[i].setData(width=p_traj_second_width.value())


def update_traj_second_colors():
    global traj_second_colors
    traj_second_colors = p_traj_second_c_map.value().getLookupTable(
        nPts=len(traj_second_plots), mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(traj_second_plots)):
        traj_second_plots[i].setData(color=traj_second_colors[i])


def update_display_traj_second():
    if p_display_traj_second.value():
        add_traj_second_plots()
        update_traj_second_width()
        update_traj_second_colors()
    else:
        clear_traj_second_plots()


def update_traj_second_point_width():
    traj_second_scatter.setData(
        pos=traj_second_points_updated, size=p_traj_second_point_width.value()
    )


def update_traj_second_point_colors():
    global traj_second_point_colors
    traj_second_point_colors = p_traj_second_c_map.value().getLookupTable(
        nPts=len(traj_second_points), mode=pg.ColorMap.FLOAT
    )
    traj_second_scatter.setData(
        pos=traj_second_points_updated,
        color=traj_second_point_colors,
        size=p_traj_second_point_width.value(),
    )


def update_time_traj_second_points():
    if p_display_traj_points_second.value():
        for i in range(len(traj_second_points_updated)):
            traj_second_points_updated[i] = point_to_cartesian(
                act_second_on_points(p_t_slider.value(), point(traj_second_points[i]))
            )
        traj_second_scatter.setData(
            pos=traj_second_points_updated,
            color=traj_second_point_colors,
            size=p_traj_second_point_width.value(),
        )


def update_display_traj_points_second():
    if p_display_traj_points_second.value():
        update_traj_second_point_width()
        update_traj_second_point_colors()
        update_time_traj_second_points()
        traj_second_scatter.setVisible(True)
    else:
        traj_second_scatter.setVisible(False)


####################
# Cube
####################


def update_cube():
    global cube_points
    global cols
    cube_points, cols = point_cube_gen(
        eval(p_cube_center.value()),
        eval(p_cube_length.value()),
        np.array(eval(p_cube_subds.value())),
    )
    full_update()


def update_display_cube():
    cube_plot.setVisible(p_display_cube.value())
    full_time_update()


####################
# Coeff
####################


def update_coeff_set():
    full_update()


def update_main_poly_coeff():
    global main_poly_coeff
    main_poly_coeff = eval(p_main_poly_coeff.value())
    full_update()


def update_first_poly_coeff():
    global first_poly_coeff
    first_poly_coeff = eval(p_first_poly_coeff.value())
    full_update()


def update_second_poly_coeff():
    global second_poly_coeff
    second_poly_coeff = eval(p_second_poly_coeff.value())
    full_update()


####################
# Points
####################


def update_display_points():
    points_scatter.setVisible(p_display_points.value())


def update_time_points():
    if p_display_points.value():
        if p_use_main_coeff.value():
            for i in range(len(points_coordinates)):
                points_current_coordinates[i] = point_to_cartesian(
                    act_main_on_points(p_t_slider.value(), point(points_coordinates[i]))
                )
        else:
            for i in range(len(points_coordinates)):
                points_current_coordinates[i] = point_to_cartesian(
                    act_factorization_on_points(
                        p_t_slider.value(), point(points_coordinates[i])
                    )
                )
        update_points_data()


def update_points_data():
    points_scatter.setData(
        pos=points_current_coordinates,
        color=points_colors,
        size=p_points_width.value(),
    )


def update_points_coordinates():
    global points_coordinates
    global points_current_coordinates
    points_coordinates = eval(p_points_params.value())
    points_current_coordinates = eval(p_points_params.value())
    update_time_points()


def update_points_colors():
    global points_colors
    points_colors = p_points_c_map.value().getLookupTable(
        nPts=len(points_coordinates), mode=pg.ColorMap.FLOAT
    )
    update_points_data()


def update_points_width():
    update_points_data()


####################
# Spheres
####################


def add_spheres():
    for sphere in view_spheres:
        view.addItem(sphere)


def remove_spheres():
    for sphere in view_spheres:
        try:
            view.removeItem(sphere)
        except ValueError:
            continue


def update_display_sphere():
    if p_display_sphere.value():
        add_spheres()
    else:
        remove_spheres()


def update_spheres():
    global view_spheres
    if p_display_sphere.value():
        remove_spheres()
    view_spheres = [
        gl.GLMeshItem(meshdata=unit_sphere, color=(0.5, 0.5, 0.5, 1))
        for i in range(len(sphere_centers))
    ]
    update_sphere_colors()
    for i in range(len(sphere_centers)):
        if p_use_main_coeff.value():
            new_sphere = act_main_on_points(
                p_t_slider.value(), sphere(sphere_centers[i], sphere_radii[i])
            )
        else:
            new_sphere = act_factorization_on_points(
                p_t_slider.value(), sphere(sphere_centers[i], sphere_radii[i])
            )
        cent, rad = sphere_to_cartesian(new_sphere)
        cent = np.real(cent)
        rad = np.real(rad)
        view_spheres[i].resetTransform()
        view_spheres[i].setMeshData(meshdata=unit_sphere)
        view_spheres[i].translate(cent[0], cent[1], cent[2])
        view_spheres[i].scale(rad, rad, rad)
    if p_display_sphere.value():
        add_spheres()


def update_sphere_params():
    global sphere_params
    global sphere_centers
    global sphere_radii
    sphere_params = eval(p_sphere_params.value())
    sphere_centers = [sphere_params[i][0] for i in range(len(sphere_params))]
    sphere_radii = [sphere_params[i][1] for i in range(len(sphere_params))]
    update_spheres()


def update_sphere_colors():
    global sphere_colors
    sphere_colors = p_sphere_c_map.value().getLookupTable(
        nPts=len(view_spheres), mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(view_spheres)):
        view_spheres[i].setColor(sphere_colors[i])


####################
# Generated Cyclic
####################


def clear_cyclic_plots():
    for i in range(len(cyclic_first_plots)):
        try:
            view.removeItem(cyclic_first_plots[i])
            view.removeItem(cyclic_second_plots[i])
        except ValueError:
            pass


def add_cyclic_plots():
    for i in range(len(cyclic_first_plots)):
        view.addItem(cyclic_first_plots[i])
        view.addItem(cyclic_second_plots[i])
        cyclic_first_plots[i].setGLOptions(blend)
        cyclic_second_plots[i].setGLOptions(blend)


def update_cyclic_width():
    for i in range(len(cyclic_first_plots)):
        cyclic_first_plots[i].setData(width=p_cyclic_width.value())
        cyclic_second_plots[i].setData(width=p_cyclic_width.value())


def update_cyclic_colors():
    global cyclic_colors
    cyclic_colors = p_cyclic_c_map.value().getLookupTable(
        nPts=2, mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(cyclic_first_plots)):
        cyclic_first_plots[i].setData(color=cyclic_colors[0])
        cyclic_second_plots[i].setData(color=cyclic_colors[1])


def update_display_cyclic():
    if p_display_cyclic.value():
        add_cyclic_plots()
        update_cyclic_width()
        update_cyclic_colors()
    else:
        clear_cyclic_plots()


def generate_cyclic_points(start_point):
    time = np.linspace(0.00, np.pi + 0.00, p_cyclic_mesh.value(), endpoint=False)
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(act_main_on_points(time[i], point(start_point)))
        )
    return points


def generate_cyclic_first_traj(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_cyclic_subd.value())
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(act_first_on_points(time[i], point(start_point)))
        )
    return points


def generate_cyclic_second_traj(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_cyclic_subd.value())
    points = np.empty((len(time), 3))
    for i in range(len(time)):
        points[i] = np.real(
            point_to_cartesian(act_second_on_points(time[i], point(start_point)))
        )
    return points


def update_cyclic():
    global cyclic_gen_point
    global cyclic_points
    global cyclic_first_plots
    global cyclic_second_plots
    cyclic_gen_point = eval(p_cyclic_params.value())
    cyclic_points = generate_cyclic_points(cyclic_gen_point)
    clear_cyclic_plots()
    cyclic_first_plots = [gl.GLLinePlotItem() for i in range(len(cyclic_points))]
    cyclic_second_plots = [gl.GLLinePlotItem() for i in range(len(cyclic_points))]
    for i in range(len(cyclic_points)):
        cyclic_first_plots[i].setData(pos=generate_cyclic_first_traj(cyclic_points[i]))
        cyclic_second_plots[i].setData(
            pos=generate_cyclic_second_traj(cyclic_points[i])
        )
    update_display_cyclic()


####################
# Custom Code
####################


def execute_custom_code():
    # Here it is NECESSARY for variables to be declared globally
    exec(p_shell_interface.value())


def update_blend():
    blend = p_blend.value()
    traj_main_scatter.setGLOptions(blend)
    traj_first_scatter.setGLOptions(blend)
    traj_second_scatter.setGLOptions(blend)
    cube_plot.setGLOptions(blend)
    update_all_params()


####################
# Updating
####################


def update_scatter_with_main():
    cube_plot.setData(
        pos=point_p_act_main_on_points(cube_points),
        color=cols,
    )


def update_scatter_with_factorization():
    cube_plot.setData(
        pos=point_p_act_factorization_on_points(cube_points),
        color=cols,
    )


def full_time_update():
    if p_display_cube.value():
        if p_use_main_coeff.value():
            update_scatter_with_main()
        else:
            update_scatter_with_factorization()
    if p_display_sphere.value():
        update_spheres()

    update_time_traj_main_points()
    update_time_traj_first_points()
    update_time_traj_second_points()
    update_time_points()


def full_update():
    full_time_update()


def update_all_params():
    update_sphere_params()
    update_main_poly_coeff()
    update_first_poly_coeff()
    update_second_poly_coeff()
    update_traj_main()
    update_traj_first()
    update_traj_second()
    update_points_coordinates()
    update_cyclic()


view.addItem(cube_plot)
view.addItem(traj_main_scatter)
traj_main_scatter.setVisible(False)
view.addItem(traj_first_scatter)
traj_first_scatter.setVisible(False)
view.addItem(traj_second_scatter)
traj_second_scatter.setVisible(False)
view.addItem(points_scatter)
points_scatter.setVisible(False)

# Connect the slider change signals to the update functions
p_update_all.sigActivated.connect(update_all_params)

p_t_slider.sigValueChanged.connect(update_t_slide)
p_t_value.sigValueChanged.connect(update_t_val)

p_execute_code.sigActivated.connect(execute_custom_code)

p_background_color.sigValueChanged.connect(update_bg_color)

p_show_axes.sigValueChanged.connect(update_show_axes)
p_axes_size.sigValueChanged.connect(update_axes_size)

p_use_main_coeff.sigValueChanged.connect(update_coeff_set)
p_update_coeff.sigActivated.connect(update_main_poly_coeff)
p_update_coeff.sigActivated.connect(update_first_poly_coeff)
p_update_coeff.sigActivated.connect(update_second_poly_coeff)


p_display_cube.sigValueChanged.connect(update_display_cube)
p_cube_subds.sigValueChanged.connect(update_cube)
p_cube_center.sigValueChanged.connect(update_cube)
p_cube_length.sigValueChanged.connect(update_cube)


p_display_traj_main.sigValueChanged.connect(update_display_traj_main)
p_display_traj_points_main.sigValueChanged.connect(update_display_traj_points_main)
p_traj_main_point_width.sigValueChanged.connect(update_traj_main_point_width)
p_traj_main_points_update.sigActivated.connect(update_traj_main)
p_traj_main_width.sigValueChanged.connect(update_traj_main_width)
p_traj_main_c_map.sigValueChanged.connect(update_traj_main_colors)
p_traj_main_c_map.sigValueChanged.connect(update_traj_main_point_colors)

p_display_traj_first.sigValueChanged.connect(update_display_traj_first)
p_display_traj_points_first.sigValueChanged.connect(update_display_traj_points_first)
p_traj_first_point_width.sigValueChanged.connect(update_traj_first_point_width)
p_traj_first_points_update.sigActivated.connect(update_traj_first)
p_traj_first_points_update.sigActivated.connect(update_traj_first)
p_traj_first_width.sigValueChanged.connect(update_traj_first_width)
p_traj_first_c_map.sigValueChanged.connect(update_traj_first_colors)
p_traj_first_c_map.sigValueChanged.connect(update_traj_first_point_colors)

p_display_traj_second.sigValueChanged.connect(update_display_traj_second)
p_traj_second_points_update.sigActivated.connect(update_traj_second)
p_display_traj_points_second.sigValueChanged.connect(update_display_traj_points_second)
p_traj_second_point_width.sigValueChanged.connect(update_traj_second_point_width)
p_traj_second_points_update.sigActivated.connect(update_traj_second)
p_traj_second_width.sigValueChanged.connect(update_traj_second_width)
p_traj_second_c_map.sigValueChanged.connect(update_traj_second_colors)
p_traj_second_c_map.sigValueChanged.connect(update_traj_second_point_colors)

p_display_points.sigValueChanged.connect(update_display_points)
p_update_points.sigActivated.connect(update_points_coordinates)
p_points_c_map.sigValueChanged.connect(update_points_colors)
p_points_width.sigValueChanged.connect(update_points_width)


p_display_sphere.sigValueChanged.connect(update_display_sphere)
p_update_spheres.sigActivated.connect(update_sphere_params)
p_sphere_c_map.sigValueChanged.connect(update_sphere_colors)


p_display_cyclic.sigValueChanged.connect(update_display_cyclic)
p_update_cyclics.sigActivated.connect(update_cyclic)
p_cyclic_width.sigValueChanged.connect(update_cyclic_width)
p_cyclic_c_map.sigValueChanged.connect(update_cyclic_colors)
# p_cyclic_params.sigValueChanged.connect(update_cyclic)
# p_cyclic_subd.sigValueChanged.connect(update_cyclic)
# p_cyclic_mesh.sigValueChanged.connect(update_cyclic)

p_blend.sigValueChanged.connect(update_blend)
# Update all parameters before first view
update_all_params()
full_update()

# show the window
win.show()


# The typical stuff for standalone execution
if __name__ == "__main__":
    pg.exec()
