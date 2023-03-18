import matplotlib.pyplot as plt
import seaborn as sb
import sys



def plot_moistures_oneSensorAtaTime(field_1, field_2, field_3, field_4, field_5, yLabel, xLabel, y_limits=[-0.02, 1.02], ):
    fig, axs = plt.subplots(2, 2, figsize=(20,12),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2), (ax3, ax4) = axs;
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True);

    ax1.plot(field_1.Date.values, field_1.Sensor1.values, label = field_1.field.unique())
    ax1.plot(field_2.Date.values, field_2.Sensor1.values, label = field_2.field.unique())
    ax1.plot(field_3.Date.values, field_3.Sensor1.values, label = field_3.field.unique())
    ax1.plot(field_4.Date.values, field_4.Sensor1.values, label = field_4.field.unique())
    ax1.plot(field_5.Date.values, field_5.Sensor1.values, label = field_5.field.unique())
    ax1.legend(loc="best", fontsize=12);
    ax1.set_title("Sensor 1 - 4\" deep");
    if len(yLabel)>0 and len(xLabel)>0:
        ax1.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax1.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax1.set(xlabel=xLabel)
    ax1.set_ylim(y_limits)


    ax2.plot(field_1.Date.values, field_1.Sensor2.values, label= field_1.field.unique())
    ax2.plot(field_2.Date.values, field_2.Sensor2.values, label= field_2.field.unique())
    ax2.plot(field_3.Date.values, field_3.Sensor2.values, label= field_3.field.unique())
    ax2.plot(field_4.Date.values, field_4.Sensor2.values, label= field_4.field.unique())
    ax2.plot(field_5.Date.values, field_5.Sensor2.values, label= field_5.field.unique())
    ax2.legend(loc="best", fontsize=12);
    ax2.set_title("Sensor 2 - 8\" deep");
    if len(yLabel)>0 and len(xLabel)>0:
        ax2.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax2.set( ylabel=yLabel)
    elif len(xLabel) > 0:
        ax2.set(xlabel=xLabel)
    ax2.set_ylim(y_limits)


    ax3.plot(field_1.Date.values, field_1.Sensor3.values, label = field_1.field.unique())
    ax3.plot(field_2.Date.values, field_2.Sensor3.values, label = field_2.field.unique())
    ax3.plot(field_3.Date.values, field_3.Sensor3.values, label = field_3.field.unique())
    ax3.plot(field_4.Date.values, field_4.Sensor3.values, label = field_4.field.unique())
    ax3.plot(field_5.Date.values, field_5.Sensor3.values, label = field_5.field.unique())
    ax3.legend(loc="best", fontsize=12);
    ax3.set_title("Sensor 3 - 12\" deep");
    # ax3.ylim = (0, 1.2)
    if len(yLabel)>0 and len(xLabel)>0:
        ax3.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax3.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax3.set(xlabel=xLabel)
    ax3.set_ylim(y_limits)


    ax4.plot(field_1.Date.values, field_1.Sensor4.values, label = field_1.field.unique())
    ax4.plot(field_2.Date.values, field_2.Sensor4.values, label = field_2.field.unique())
    ax4.plot(field_3.Date.values, field_3.Sensor4.values, label = field_3.field.unique())
    ax4.plot(field_4.Date.values, field_4.Sensor4.values, label = field_4.field.unique())
    ax4.plot(field_5.Date.values, field_5.Sensor4.values, label = field_5.field.unique())
    ax4.legend(loc="best", fontsize=12);
    ax4.set_title("Sensor 4 - 20\" deep");
    # ax4.ylim = (0, 1.2)
    if len(yLabel)>0 and len(xLabel)>0:
        ax4.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax4.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax4.set(xlabel=xLabel)
    ax4.set_ylim(y_limits)

    # A3.plot(x='Date', y = 'Sensor1', legend = "Sensor1", ax=ax1);
    # axs[0].bar(names, values)
    # axs[1].scatter(names, values)
    # axs[2].plot(names, values)
    # fig.suptitle('Title here');


def plot_moistures_oneFieldAtaTime(field_1, field_2, field_3, field_4, field_5, yLabel="", xLabel="", y_limits=[-0.02, 1.02]):
    fig, axs = plt.subplots(3, 2, figsize=(20,12),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.1, 'wspace': .1});

    (ax1, ax2), (ax3, ax4), (ax5, ax6) = axs;
    ax1.grid(True); ax2.grid(True); ax3.grid(True); ax4.grid(True); ax5.grid(True); ax6.grid(True);

    ax1.plot(field_1.Date.values, field_1.Sensor1.values, label="Sensor 1")
    ax1.plot(field_1.Date.values, field_1.Sensor2.values, label="Sensor 2")
    ax1.plot(field_1.Date.values, field_1.Sensor3.values, label="Sensor 3")
    ax1.plot(field_1.Date.values, field_1.Sensor4.values, label="Sensor 4")
    ax1.legend(loc="best", fontsize=12);
    ax1.set_title(field_1.field.unique());
    if len(yLabel)>0 and len(xLabel)>0:
        ax1.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax1.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax1.set(xlabel=xLabel)
    ax1.set_ylim(y_limits)

    ax2.plot(field_2.Date.values, field_2.Sensor1.values, label="Sensor 1")
    ax2.plot(field_2.Date.values, field_2.Sensor2.values, label="Sensor 2")
    ax2.plot(field_2.Date.values, field_2.Sensor3.values, label="Sensor 3")
    ax2.plot(field_2.Date.values, field_2.Sensor4.values, label="Sensor 4")
    ax2.legend(loc="best", fontsize=12);

    ax2.set_title(field_2.field.unique());
    if len(yLabel)>0 and len(xLabel)>0:
        ax2.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax2.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax2.set(xlabel=xLabel)
    ax2.set_ylim(y_limits)

    ax3.plot(field_3.Date.values, field_3.Sensor1.values, label="Sensor 1")
    ax3.plot(field_3.Date.values, field_3.Sensor2.values, label="Sensor 2")
    ax3.plot(field_3.Date.values, field_3.Sensor3.values, label="Sensor 3")
    ax3.plot(field_3.Date.values, field_3.Sensor4.values, label="Sensor 4")
    ax3.legend(loc="best", fontsize=12);
    ax3.set_title(field_3.field.unique());
    if len(yLabel)>0 and len(xLabel)>0:
        ax3.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax3.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax3.set(xlabel=xLabel)
    ax3.set_ylim(y_limits)

    ax4.plot(field_4.Date.values, field_4.Sensor1.values, label="Sensor 1")
    ax4.plot(field_4.Date.values, field_4.Sensor2.values, label="Sensor 2")
    ax4.plot(field_4.Date.values, field_4.Sensor3.values, label="Sensor 3")
    ax4.plot(field_4.Date.values, field_4.Sensor4.values, label="Sensor 4")
    ax4.legend(loc="best", fontsize=12);
    ax4.set_title(field_4.field.unique());
    if len(yLabel)>0 and len(xLabel)>0:
        ax4.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax4.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax4.set(xlabel=xLabel)
    ax4.set_ylim(y_limits)

    ax5.plot(field_5.Date.values, field_5.Sensor1.values, label = "Sensor 1")
    ax5.plot(field_5.Date.values, field_5.Sensor2.values, label = "Sensor 2")
    ax5.plot(field_5.Date.values, field_5.Sensor3.values, label = "Sensor 3")
    ax5.plot(field_5.Date.values, field_5.Sensor4.values, label = "Sensor 4")
    ax5.legend(loc="best", fontsize=12);
    ax5.set_title(field_5.field.unique());
    ax5.set_ylim(y_limits)
    if len(yLabel)>0 and len(xLabel)>0:
        ax5.set(xlabel=xLabel, ylabel=yLabel)
    elif len(yLabel) > 0:
        ax5.set(ylabel=yLabel)
    elif len(xLabel) > 0:
        ax5.set(xlabel=xLabel)

