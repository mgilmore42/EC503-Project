import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_scatter_vs_time(metrics, set_name):
    algos = ['grad_boost', 'log_reg', 'rand_forest', 'kernel_svm', 'neural_net']
    symbols = ['o', '^', 's', 'D', '*']  # Circle, Triangle, Square, Diamond, Star

    # Extract training times from the 'Training Time' row in metrics
    training_time_row = metrics[-1]  # Assuming the last row contains training times
    times = np.array(training_time_row[1:], dtype=float)  # Convert values to float

    adjusted_palette = sns.color_palette("colorblind", len(algos))

    nrows, ncols = 3, 2  # Set the layout to 2 rows and 3 columns

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    for i, metric_row in enumerate(metrics[:-1]):  # Exclude the training time row
        ax = axes[i // ncols, i % ncols]
        metric_name = metric_row[0]  # First element is the metric name
        metric_values = np.array(metric_row[1:], dtype=float)  # Convert values to float

        # Plot each algorithm's metric value
        for j, value in enumerate(metric_values):
            ax.scatter(times[j], value, color=adjusted_palette[j], marker='o', s=50)

        ax.set_xlabel('Training time (s)')
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)  # Adjust this if your metrics have a range outside [0, 1]
        ax.set_title(f'{set_name} Dataset: {metric_name} vs. Training Time')

    # Remove the last subplot (lower right) and use it for the legend
    fig.delaxes(axes[2, 1])

    # Creating legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color=adjusted_palette[i], linestyle='None', markersize=10)
                      for i in range(len(algos))]

    # Place the legend in the empty subplot space
    fig.legend(handles=legend_handles, labels=algos, title='Algorithm', loc='lower right')

    plt.tight_layout()
    plt.show()

# Scatter plots of the testing performance vs time
metrics_heart = np.array([['Test CCR', 0.8032786885, 0.8524590164, 0.8360655738,
                           0.868852459, 0.8032786885],
                          ['Train CCR', 0.9669421488, 0.8636363636, 1.0,
                           0.9008264463, 0.9917355372],
                          ['Test Precision', 0.8172221221, 0.8530759739, 0.8360655738,
                           0.868852459, 0.8052176979],
                          ['Test Recall', 0.8032786885, 0.8524590164, 0.8360655738,
                           0.868852459, 0.8032786885],
                          ['F1 score', 0.8023247598, 0.8525384035,
                           0.8360655738, 0.868852459, 0.8033844527],
                          ['Training Time', 0.020891708, 0.001876833994, 0.04850262501, 0.001386666001, 0.131930875]],
                         dtype=object)
plot_scatter_vs_time(metrics_heart, 'Heart')

metrics_rain = np.array([['Test CCR', 0.8533982732, 0.8469780828, 0.8592871375,
                           0.8468452513, 0.8516271862],
                          ['Train CCR', 0.8573500111, 0.8494908125, 1.0,
                           0.8491144565, 0.8536196591],
                          ['Test Precision', 0.845000932, 0.8373489811, 0.8521155937,
                           0.8376149016, 0.8428508405],
                          ['Test Recall', 0.8533982732, 0.8469780828, 0.8592871375,
                           0.8468452513, 0.8516271862],
                          ['F1 score', 0.842728139, 0.8358667586,
                           0.8487640881, 0.8333425246, 0.8421781878],
                          ['Training Time', 23.95534354, 1.214116292, 63.86219158, 355.3854349, 74.34482008]],
                         dtype=object)
plot_scatter_vs_time(metrics_rain, 'Rain')

metrics_campus = np.array([['Test CCR', 0.8, 0.8333333333, 0.9333333333,
                           0.8666666667, 0.8],
                          ['Train CCR', 0.9237288136, 0.8474576271, 1.0,
                           0.8559322034, 1.0],
                          ['Test Precision', 0.8619047619, 0.8447204969, 0.95,
                           0.8886363636, 0.86],
                          ['Test Recall', 0.8, 0.8333333333, 0.9333333333,
                           0.8666666667, 0.8],
                          ['F1 score', 0.8253968254, 0.8379705401,
                           0.9366459627, 0.8732919255, 0.8159090909],
                          ['Training Time', 0.058005916, 0.003907292004, 0.043336, 0.001146124996, 0.03202095801]],
                         dtype=object)
plot_scatter_vs_time(metrics_campus, 'Campus')

metrics_housing = np.array([['Test CCR', 0.07, 0.09, 0.085,
                           0.09, 0.06],
                          ['Train CCR', 0.34875, 0.14, 1.0,
                           0.99875, 0.1325],
                          ['Test Precision', 0.06762998801, 0.09285033235, 0.08287226721,
                           0.0839122807, 0.01174074074],
                          ['Test Recall', 0.07, 0.09, 0.085,
                           0.09, 0.06],
                          ['F1 score', 0.06836035708, 0.07798500051,
                           0.08296098906, 0.05717164824, 0.01884719156],
                          ['Training Time', 0.499722791, 0.070728958, 0.129216459, 0.02198275001, 0.107878458]],
                         dtype=object)
plot_scatter_vs_time(metrics_housing, 'Housing')