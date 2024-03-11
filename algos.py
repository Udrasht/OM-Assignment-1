from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os


dict_steapest_desent = {
    "Backtracking": {
        "alptha_0": 10.0,
        "rho": 0.75,
        "k": 0,
        "epsilon": .000001,
        "max_iter": 10000,
        "c": 0.001

    },
    "Bisection": {
        "c1": 0.001,
        "c2": 0.1,
        "alpha": 0,
        "t": 1,
        "beta": 1000000,
        "k": 0,
        "epsilon": .000001,
        "max_iter": 10000,

    }
}
dict_newton_methods = {
    "max_iter": 10000,
    "epsilon": .000001,
    "Damped": {
        "alpha": 0.001,
        "beta": 0.75

    }



}


def plot_fx_vs_iterations(
    fx_val: list[float],
    k: int,
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection","Pure", "Damped", "Levenberg-Marquardt", "Combined"],
    x_axis_lable: str,
    y_axis_lable: str,
    graph_type: str
):
    plt.figure(figsize=(12, 5))
    plt.plot(range(0, k), fx_val, marker='o')
    plt.xlabel(x_axis_lable)
    plt.ylabel(y_axis_lable)
    plt.title(f'{f.__name__} vs iteration of {condition}_{graph_type}')

    # Check if the plot folder exists, if not create it
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot in the plots folder
    plt.savefig(f'plots/{f.__name__}_{np.array2string(initial_point)}_{condition}_{graph_type}.png')
    plt.close()


def plot_contour(condition: Literal["Backtracking", "Bisection","Pure", "Damped", "Levenberg-Marquardt", "Combined"],
                 f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
                 inital_point: npt.NDArray[np.float64],
                 coordinates: list[float],
                 ):
    array = np.array(coordinates)

    Z_cord = []

   
    stepx=(np.round(np.max(array[:,0]))+.4-(np.round(np.min(array[:,0]))-.4))/50.0
    stepy=(np.round(np.max(array[:,1]))+.4-(np.round(np.min(array[:,1]))-.4))/50.0
    
    arrayx= np.arange(np.round(np.min(array[:,0]))-.4, np.round(np.max(array[:,0]))+.4, stepx)
    

    x=arrayx
    
    arrayy= np.arange(np.round(np.min(array[:,1]))-.4, np.round(np.max(array[:,1]))+.4, stepy)
    y=arrayy

    
    
    
    

    X, Y = np.meshgrid( x,  y)
    
    n1 = len(X)
    n2 = len(Y)
    

    for i in range(n1):

        temp = []
        for j in range(n2):
            val = f(np.array([X[i][j], Y[i][j]]))
            temp.append(val)
        Z_cord.append(temp)
    plt.figure(figsize=(20,10))
    plt.contour(X, Y, np.array(Z_cord),levels=20)
    # array=np.concatenate((array[:60], array[-60:]))
    for i in range(len(array)-1):
        plt.arrow(array[i][0], array[i][1], array[i+1][0]-array[i][0], array[i+1]
                  [1]-array[i][1], head_width=0.020, head_length=0.025, fc='cyan', ec='orange')

    # Mark the initial point with yellow dot
    plt.scatter(array[0][0], array[0][1], color='black', label='Initial Point')
    plt.scatter(array[-1][0], array[-1][1], color='red', label='Final Point')
    plt.xlabel('X_axis')
    plt.ylabel('Y_axis')
    plt.title(f'Contour Plot with Update Arrows ({f.__name__})')
    plt.colorbar()
    plt.legend()
    plt.savefig(
        f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
    plt.close()


   


# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    # print(condition,"up",f)
    X_y_cord = []

    X_y_cord.append(inital_point)
    if condition == "Backtracking":
        alpha_0 = dict_steapest_desent[condition]["alptha_0"]
        k = dict_steapest_desent[condition]["k"]
        epsilon = dict_steapest_desent[condition]["epsilon"]
        rho = dict_steapest_desent[condition]["rho"]
        maxIteration = dict_steapest_desent[condition]["max_iter"]
        c = dict_steapest_desent[condition]["c"]
        X = inital_point

        fx_val = []
        fx_grad_norm_val = []
        while k < maxIteration:
            gradiant = d_f(X)

            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                alpha = alpha_0

                dk = -gradiant
                f_x = f(X)
                updated_X = X+alpha*dk

                f_x_alpha_d = f(updated_X)
               

                while f_x_alpha_d > f_x+c*alpha*np.dot(gradiant, dk):
                    alpha = alpha*rho
                    updated_X = X+alpha*dk
                    f_x_alpha_d = f(updated_X)
                X = updated_X
                X_y_cord.append(X)

                k = k+1
                

            else:
                break
       

        x_axis_lable = 'Iterations'
        y_axis_lable = 'f(x)'
        graph_type = 'vals'
        plot_fx_vs_iterations(fx_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)
        y_axis_lable = "|f'(x)|"
        graph_type = 'grad'
        plot_fx_vs_iterations(fx_grad_norm_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)

        if len(X_y_cord[0]) == 2:

            plot_contour(condition, f, inital_point, X_y_cord)
        print(k,condition,inital_point,f)
        return X

    if condition == "Bisection":
        c1 = dict_steapest_desent[condition]["c1"]
        c2 = dict_steapest_desent[condition]["c2"]
        alpha_0 = dict_steapest_desent[condition]["alpha"]
        t = dict_steapest_desent[condition]["t"]
        beta_0 = dict_steapest_desent[condition]["beta"]
        k = dict_steapest_desent[condition]["k"]
        epsilon = dict_steapest_desent[condition]["epsilon"]
        maxIteration = dict_steapest_desent[condition]["max_iter"]

        X = inital_point
        fx_val = []
        fx_grad_norm_val = []
        while k < maxIteration:
            gradiant = d_f(X)
            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                alpha = alpha_0
                beta = beta_0

                dk = -gradiant

                while (True):
                    if f(X+t*dk) > f(X)+c1*t*(np.dot(gradiant, dk)):
                        beta = t
                        t = (alpha+beta)/2
                    elif np.dot(d_f(X+t*dk), dk) < c2*(np.dot(gradiant, dk)):
                        alpha = t
                        t = (alpha+beta)/2
                    else:
                        break
                X = X+t*dk
                X_y_cord.append(X)
                k = k+1
            else:
                break
        x_axis_lable = 'Iterations'
        y_axis_lable = 'f(x)'
        graph_type = 'vals'
        plot_fx_vs_iterations(fx_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)
        y_axis_lable = "|f'(x)|"
        graph_type = 'grad'
        plot_fx_vs_iterations(fx_grad_norm_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)
        if len(X_y_cord[0]) == 2:

            plot_contour(condition, f, inital_point, X_y_cord)
        print(k,condition,inital_point,f)
        return X

   


# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    
    max_iteration = dict_newton_methods["max_iter"]
    epsilon = dict_newton_methods["epsilon"]
    k = 0
    X = inital_point
    fx_val = []
    fx_grad_norm_val = []
    
    X_y_cord = []

    X_y_cord.append(inital_point)

    if condition == "Pure":
        while k < max_iteration:
            gradiant = d_f(X)
            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                dk = np.linalg.solve(d2_f(X), gradiant)
                X = X-dk
                X_y_cord.append(X)
                k = k+1
            else:
                break
    elif condition == "Damped":
        alpha = dict_newton_methods[condition]["alpha"]
        beta = dict_newton_methods[condition]["beta"]

        while k < max_iteration:
            gradiant = d_f(X)
            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                t = 1
                dk = -np.linalg.solve(d2_f(X), gradiant)
                while f(X) < f(X+t*dk)-alpha*t*(np.dot(gradiant, dk)):
                    t = beta*t
                X = X+t*dk
                X_y_cord.append(X)
                k = k+1
            else:
                break
    elif condition == "Levenberg-Marquardt":
    

        while k < max_iteration:
            gradiant = d_f(X)
            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                hessian = d2_f(X)
                lambda_min = min(np.linalg.eigvals(hessian))
                if lambda_min <= 0:
                    mue_k = -lambda_min+0.1
                    temp = np.linalg.inv(hessian + mue_k * np.eye(len(X)))
                    dk = -np.dot(temp, gradiant)
                else:
                    dk = -np.linalg.solve(d2_f(X), gradiant)
                X = X+dk
                X_y_cord.append(X)
                k = k+1
            else:
                break
    elif condition == "Combined":

        
        rho = 0.75
        c = 0.001

        while k < max_iteration:
            gradiant = d_f(X)
            if np.linalg.norm(gradiant) > epsilon:
                fx_val.append(f(X))
                fx_grad_norm_val.append(np.linalg.norm(gradiant))
                hessian = d2_f(X)
                lambda_min = min(np.linalg.eigvals(hessian))
                if lambda_min <= 0:
                    mue_k = -lambda_min+0.1
                    temp = np.linalg.inv(hessian + mue_k * np.eye(len(X)))
                    dk = -np.dot(temp, gradiant)
                else:
                    dk = -np.linalg.solve(d2_f(X), gradiant)
                alpha = 10.0
                updated_X = X+alpha*dk
                f_x_alpha_d = f(updated_X)
                f_x = f(X)
                while f_x_alpha_d > f_x+c*alpha*np.dot(gradiant, dk):
                    alpha = alpha*rho
                    updated_X = X+alpha*dk
                    f_x_alpha_d = f(updated_X)

                X = updated_X
                X_y_cord.append(X)
                k = k+1
            else:
                break
    x_axis_lable = 'Iterations'
    y_axis_lable = 'f(x)'
    graph_type = 'vals'
    plot_fx_vs_iterations(fx_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)
    y_axis_lable = "|f'(x)|"
    graph_type = 'grad'
    plot_fx_vs_iterations(fx_grad_norm_val, k, f, inital_point,condition, x_axis_lable, y_axis_lable, graph_type)
    if len(X_y_cord[0]) == 2:

            plot_contour(condition, f, inital_point, X_y_cord)
    print(k,condition,inital_point,f)
    return X

    
