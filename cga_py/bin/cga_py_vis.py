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
p_t_slider.setLimits([-np.pi, np.pi])
p_t_slider.setOpts(step=np.pi / 300)
p_t_slider.setValue(np.pi / 2 + 0.001)
p_t_slider.setDefault(np.pi / 2 + 0.001)
# Add a slider
p_t_value = time_parameters.addChild(
    pTypes.SimpleParameter(type="str", name="Time Input")
)
p_t_value.setValue(p_t_slider.value())
p_t_value.setDefault(p_t_slider.value())


# What happens at update of parameter
def update_t_slide():
    # p_t_value.setValue(p_t_slider.value())
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
p_shell_interface = general_params.addChild(pTypes.TextParameter(name="Custom Code"))
p_shell_interface.setValue(
    "#This Field can execute custom code. Variables must be declared with the global keyword!"
)

axes_tree = general_params.addChild(Parameter.create(name="Axes", type="group"))
p_show_axes = axes_tree.addChild(pTypes.SimpleParameter(name="Show Axes", type="bool"))
p_show_axes.setValue(True)
p_show_axes.setDefault(True)
p_axes_size = axes_tree.addChild(
    pTypes.SimpleParameter(name="Axes Lengths", type="str")
)
p_axes_size.setValue("[3,3,3]")
p_axes_size.setDefault("[3,3,3]")


# Polynomial input
poly_tree = general_params.addChild(Parameter.create(name="Polynomial", type="group"))

p_update_coeff = poly_tree.addChild(pTypes.ActionParameter(name="Update Coefficients"))


p_use_main_coeff = poly_tree.addChild(
    pTypes.SimpleParameter(name="Use Main Polynomial", type="bool")
)
p_use_main_coeff.setValue(True)


p_main_poly_coeff = poly_tree.addChild(pTypes.TextParameter(name="Main Coeff"))
p_main_poly_coeff.setValue("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")
p_main_poly_coeff.setDefault("[0.5*e_123o - e_123i, e_12 - e_3i + 0.5*e_3o, 1]")

x = 1
y = 1
z = 100

p_first_poly_coeff = poly_tree.addChild(pTypes.TextParameter(name="First Coeff"))
p_first_poly_coeff.setValue(
    "[-1/2*(2*x**4*e_3i-x**4*e_3o+4*x**2*y**2*e_3i-2*x**2*y**2*e_3o+4*x**2*z**2*e_3i-2*x**2*z**2*e_3o+2*y**4*e_3i-y**4*e_3o+4*y**2*z**2*e_3i-2*y**2*z**2*e_3o+2*z**4*e_3i-z**4*e_3o-8*x**3*e_13+8*x**3*e_2i-4*x**3*e_2o-8*x**2*y*e_1i+4*x**2*y*e_1o-8*x**2*y*e_23-8*x*y**2*e_13+8*x*y**2*e_2i-4*x*y**2*e_2o-8*x*z**2*e_13+8*x*z**2*e_2i-4*x*z**2*e_2o-8*y**3*e_1i+4*y**3*e_1o-8*y**3*e_23-8*y*z**2*e_1i+4*y*z**2*e_1o-8*y*z**2*e_23-32*x**2*e_12-16*x**2*e_3i+8*x**2*e_3o+32*x*z*e_1i-16*x*z*e_1o+32*x*z*e_23-32*y**2*e_12-16*y**2*e_3i+8*y**2*e_3o-32*y*z*e_13+32*y*z*e_2i-16*y*z*e_2o+16*z**2*e_3i-8*z**2*e_3o+32*x*e_13-32*x*e_2i+16*x*e_2o+32*y*e_1i-16*y*e_1o+32*y*e_23+32*e_3i-16*e_3o)/(x**2+y**2+z**2+4)**2, 1]"
)
p_first_poly_coeff.setDefault(
    "[-1/2*(2*x**4*e_3i-x**4*e_3o+4*x**2*y**2*e_3i-2*x**2*y**2*e_3o+4*x**2*z**2*e_3i-2*x**2*z**2*e_3o+2*y**4*e_3i-y**4*e_3o+4*y**2*z**2*e_3i-2*y**2*z**2*e_3o+2*z**4*e_3i-z**4*e_3o-8*x**3*e_13+8*x**3*e_2i-4*x**3*e_2o-8*x**2*y*e_1i+4*x**2*y*e_1o-8*x**2*y*e_23-8*x*y**2*e_13+8*x*y**2*e_2i-4*x*y**2*e_2o-8*x*z**2*e_13+8*x*z**2*e_2i-4*x*z**2*e_2o-8*y**3*e_1i+4*y**3*e_1o-8*y**3*e_23-8*y*z**2*e_1i+4*y*z**2*e_1o-8*y*z**2*e_23-32*x**2*e_12-16*x**2*e_3i+8*x**2*e_3o+32*x*z*e_1i-16*x*z*e_1o+32*x*z*e_23-32*y**2*e_12-16*y**2*e_3i+8*y**2*e_3o-32*y*z*e_13+32*y*z*e_2i-16*y*z*e_2o+16*z**2*e_3i-8*z**2*e_3o+32*x*e_13-32*x*e_2i+16*x*e_2o+32*y*e_1i-16*y*e_1o+32*y*e_23+32*e_3i-16*e_3o)/(x**2+y**2+z**2+4)**2, 1]"
)

p_second_poly_coeff = poly_tree.addChild(pTypes.TextParameter(name="Second Coeff"))
p_second_poly_coeff.setValue(
    "[-16/(x**2+y**2+z**2+4)**2*e_12*x**2-16/(x**2+y**2+z**2+4)**2*e_12*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x**3+16/(x**2+y**2+z**2+4)**2*e_13*x-4/(x**2+y**2+z**2+4)**2*e_1i*y**3+16/(x**2+y**2+z**2+4)**2*e_1i*y+2/(x**2+y**2+z**2+4)**2*e_1o*y**3-8/(x**2+y**2+z**2+4)**2*e_1o*y-4/(x**2+y**2+z**2+4)**2*e_23*y**3+16/(x**2+y**2+z**2+4)**2*e_23*y+4/(x**2+y**2+z**2+4)**2*e_2i*x**3-16/(x**2+y**2+z**2+4)**2*e_2i*x-2/(x**2+y**2+z**2+4)**2*e_2o*x**3+8/(x**2+y**2+z**2+4)**2*e_2o*x+16/(x**2+y**2+z**2+4)**2*e_3i+1/(x**2+y**2+z**2+4)**2*e_3i*x**4+1/(x**2+y**2+z**2+4)**2*e_3i*y**4+1/(x**2+y**2+z**2+4)**2*e_3i*z**4-8/(x**2+y**2+z**2+4)**2*e_3i*x**2-8/(x**2+y**2+z**2+4)**2*e_3i*y**2+8/(x**2+y**2+z**2+4)**2*e_3i*z**2-8/(x**2+y**2+z**2+4)**2*e_3o-1/2/(x**2+y**2+z**2+4)**2*e_3o*x**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*y**4-1/2/(x**2+y**2+z**2+4)**2*e_3o*z**4+4/(x**2+y**2+z**2+4)**2*e_3o*x**2+4/(x**2+y**2+z**2+4)**2*e_3o*y**2-4/(x**2+y**2+z**2+4)**2*e_3o*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*y**2+2/(x**2+y**2+z**2+4)**2*e_3i*x**2*z**2+2/(x**2+y**2+z**2+4)**2*e_3i*z**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*y**2-1/(x**2+y**2+z**2+4)**2*e_3o*x**2*z**2-1/(x**2+y**2+z**2+4)**2*e_3o*z**2*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*y**2-4/(x**2+y**2+z**2+4)**2*e_13*x*z**2-16/(x**2+y**2+z**2+4)**2*e_13*y*z-4/(x**2+y**2+z**2+4)**2*e_1i*x**2*y-4/(x**2+y**2+z**2+4)**2*e_1i*y*z**2+16/(x**2+y**2+z**2+4)**2*e_1i*x*z+2/(x**2+y**2+z**2+4)**2*e_1o*x**2*y+2/(x**2+y**2+z**2+4)**2*e_1o*y*z**2-8/(x**2+y**2+z**2+4)**2*e_1o*x*z-4/(x**2+y**2+z**2+4)**2*e_23*x**2*y-4/(x**2+y**2+z**2+4)**2*e_23*y*z**2+16/(x**2+y**2+z**2+4)**2*e_23*x*z+4/(x**2+y**2+z**2+4)**2*e_2i*x*y**2+4/(x**2+y**2+z**2+4)**2*e_2i*x*z**2+16/(x**2+y**2+z**2+4)**2*e_2i*y*z-2/(x**2+y**2+z**2+4)**2*e_2o*x*y**2-2/(x**2+y**2+z**2+4)**2*e_2o*x*z**2-8/(x**2+y**2+z**2+4)**2*e_2o*y*z+e_12-e_3i+1/2*e_3o, 1]"
)


####################
# Trajectory Parameters
####################
traj_tree = general_params.addChild(Parameter.create(name="Trajectories", type="group"))

p_traj_points_update = traj_tree.addChild(
    pTypes.ActionParameter(name="Update Trajectories")
)

p_display_tajectory = traj_tree.addChild(
    pTypes.SimpleParameter(name="Display Trajectories", type="bool")
)
p_display_tajectory.setValue(False)

p_traj_points = traj_tree.addChild(pTypes.TextParameter(name="Trajectory Points"))
p_traj_points.setValue("[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20)]")
p_traj_points.setDefault("[[np.sin(t),np.cos(t),0] for t in np.linspace(0,2*np.pi,20)]")

p_traj_subds = traj_tree.addChild(
    pTypes.SimpleParameter(type="int", name="Trajectory Subdivisions")
)
p_traj_subds.setValue(100)
p_traj_subds.setDefault(100)

p_traj_width = traj_tree.addChild(
    pTypes.SimpleParameter(type="float", name="Trajectory width")
)
p_traj_width.setValue(2)
p_traj_width.setDefault(2)


p_traj_c_map = traj_tree.addChild(pTypes.ColorMapParameter(name="Trajectory Colors"))


####################
# Cube Parameters
####################
cube_tree = general_params.addChild(Parameter.create(name="Cube", type="group"))
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
# Sphere Parameters
####################
sphere_tree = general_params.addChild(Parameter.create(name="Sphere", type="group"))
p_update_spheres = sphere_tree.addChild(pTypes.ActionParameter(name="Update Spheres"))
p_display_sphere = sphere_tree.addChild(
    pTypes.SimpleParameter(name="Display Sphere", type="bool")
)
p_display_sphere.setValue(False)
p_sphere_params = sphere_tree.addChild(pTypes.TextParameter(name="Center"))
p_sphere_params.setValue(
    "[[[1, 0, 0],0.5],[[0, -1, 0],0.5],[[-1, 0, 0],0.5],[[0, 1, 0],0.5]]"
)
p_sphere_params.setDefault(
    "[[[1, 0, 0],0.5],[[0, -1, 0],0.5],[[-1, 0, 0],0.5],[[0, 1, 0],0.5]]"
)
p_sphere_c_map = sphere_tree.addChild(pTypes.ColorMapParameter(name="Sphere Colors"))


# Create a parameter tree for displaying the slider
general_paramtree = ParameterTree()
general_paramtree.setParameters(general_params, showTop=False)

main_poly_coeff = eval(p_main_poly_coeff.value())
first_poly_coeff = eval(p_first_poly_coeff.value())
second_poly_coeff = eval(p_second_poly_coeff.value())
cube_points, cols = point_cube_gen(
    eval(p_cube_center.value()), eval(p_cube_length.value()), eval(p_cube_subds.value())
)
traj_points = eval(p_traj_points.value())
traj_plots = [gl.GLLinePlotItem(), gl.GLLinePlotItem()]
traj_colors = p_traj_c_map.value().getLookupTable(
    nPts=len(traj_plots), mode=pg.ColorMap.QCOLOR
)
unit_sphere = gl.MeshData.sphere(20, 20)
sphere_params = eval(p_sphere_params.value())
sphere_centers = [sphere_params[i][0] for i in range(len(sphere_params))]
sphere_radii = [sphere_params[i][1] for i in range(len(sphere_params))]
view_spheres = [gl.GLMeshItem(meshdata=unit_sphere, color=(0.5, 0.5, 0.5, 1))]
sphere_colors = p_sphere_c_map.value().getLookupTable(
    nPts=len(view_spheres), mode=pg.ColorMap.QCOLOR
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
scatter = gl.GLScatterPlotItem(pos=cube_points, color=cols)


def update_show_axes():
    if p_show_axes.value():
        axes.setVisible(True)
    else:
        axes.setVisible(False)


def update_axes_size():
    scale = eval(p_axes_size.value())
    axes.setSize(scale[0], scale[1], scale[2])


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


def act_factorization_on_points(param, actpoint):
    return poly_act(
        np.tan(param),
        second_poly_coeff,
        poly_act(np.tan(param), first_poly_coeff, actpoint),
    )


def update_scatter_with_main():
    scatter.setData(
        pos=point_p_act_main_on_points(cube_points),
        color=cols,
    )


def update_scatter_with_factorization():
    scatter.setData(
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


def update_display_cube():
    if p_display_cube.value():
        scatter.setVisible(True)
    else:
        scatter.setVisible(False)
    full_time_update()


def generatre_trajectory_points(start_point):
    time = np.linspace(0.01, np.pi + 0.01, p_traj_subds.value())
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
    update_display_trajectory()


def update_trajectory_width():
    for i in range(len(traj_plots)):
        traj_plots[i].setData(width=p_traj_width.value())


def update_trajectory_colors():
    global traj_colors
    traj_colors = p_traj_c_map.value().getLookupTable(
        nPts=len(traj_plots), mode=pg.ColorMap.QCOLOR
    )
    for i in range(len(traj_plots)):
        traj_plots[i].setData(color=traj_colors[i])


def update_display_trajectory():
    if p_display_tajectory.value():
        add_traj_plots()
        update_trajectory_width()
        update_trajectory_colors()
    else:
        clear_traj_plots()


def update_cube():
    global cube_points
    global cols
    cube_points, cols = point_cube_gen(
        eval(p_cube_center.value()),
        eval(p_cube_length.value()),
        np.array(eval(p_cube_subds.value())),
    )
    full_update()


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


def execute_custom_code():
    # Here it is NECESSARY for variables to be declared globally
    exec(p_shell_interface.value())


def full_update():
    full_time_update()


def update_all_params():
    update_sphere_params()
    update_main_poly_coeff()
    update_first_poly_coeff()
    update_second_poly_coeff()
    update_trajectories()


view.addItem(scatter)

# Connect the slider change signals to the update functions
p_update_all.sigActivated.connect(update_all_params)

p_t_slider.sigValueChanged.connect(update_t_slide)
p_t_value.sigValueChanged.connect(update_t_val)

p_execute_code.sigActivated.connect(execute_custom_code)

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


p_display_tajectory.sigValueChanged.connect(update_display_trajectory)
p_traj_points_update.sigActivated.connect(update_trajectories)
p_traj_width.sigValueChanged.connect(update_trajectory_width)
p_traj_c_map.sigValueChanged.connect(update_trajectory_colors)

p_display_sphere.sigValueChanged.connect(update_display_sphere)
p_update_spheres.sigActivated.connect(update_sphere_params)
p_sphere_c_map.sigValueChanged.connect(update_sphere_colors)


# Update all parameters before first view
update_all_params()
full_update()

# show the window
win.show()


# The typical stuff for standalone execution
if __name__ == "__main__":
    pg.exec()
