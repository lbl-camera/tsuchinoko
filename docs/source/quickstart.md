# Getting Started

This is a quick-start guide that will help you install Tsuchinoko
and explore a simulated adaptive experiment.

For more in-depth documentation for developing custom adaptive experiments see:

* [AdaptiveEngine documentation](api/adaptiveengine.md)
* [ExecutionEngine documentation](api/executionengine.md)
* [Core service documentation](api/core.md)

## Install Tsuchinoko

To begin, you will need to install [Python 3.12](https://www.python.org/downloads/release/python-3120/). It is generally
recommended that you next create a [virtual environment](https://docs.python.org/3/library/venv.html) to contain the 
installation.

```console
$ python -m venv tsuchinoko-venv
```

You will then need to activate the virtual environment. This varies by operating system.

| Platform | Shell           | Command to activate virtual environment |
|----------|-----------------|-----------------------------------------|
| POSIX    | bash/zsh        | $ source tsuchinoko-venv/bin/activate            |
|          | fish            | $ source tsuchinoko-venv/bin/activate.fish       |
|          | csh/tcsh        | $ source tsuchinoko-venv/bin/activate.csh        |
|          | PowerShell Core | $ tsuchinoko-venv/bin/Activate.ps1               |
| Windows  | cmd.exe         | C:\> tsuchinoko-venv\Scripts\activate.bat        |
|          | PowerShell      | PS C:\> tsuchinoko-venv\Scripts\Activate.ps1     |

With the virtual environment active, you will then install Tsuchinoko:

```console
$ pip install tsuchinoko
```

Tsuchinoko should now be installed! You can run the Tsuchinoko client to quickly test the installation. Note that a
running Tsuchinoko server will be needed to run any experiment.

```console
$ tsuchinoko
```

```{image} _static/startup-connecting.PNG
:alt: Tsuchinoko with no connection
:width: 500px
:align: center
```

According to your preferences, components of Tsuchinoko can also be distributed across multiple systems to accommodate a 
distributed design which leverages different hardware resources. If you plan to do this, you'll need a Tsuchinoko 
installation on each system.

Did something go wrong? See [Installation Troubleshooting](installation_troubleshooting).

## Running Tsuchinoko with a Simulated Experiment

Let's try out a simulated adaptive experiment now! In this example, Tsuchinoko will adaptively sample data from a source image to create a
reconstruction by simulated measurements. You'll need the [server example](https://github.com/lbl-camera/tsuchinoko/blob/master/examples/server_demo.py) 
script and an [image](https://raw.githubusercontent.com/lbl-camera/tsuchinoko/master/examples/example_assets/sombrero_pug.jpg) 
to be sampled from. Download both of these files. We'll discuss the contents of the example server script later.

With the virtual environment active and both these files in the current directory, start the Tsuchinoko core server:

```console
$ python server_demo.py
```

The core server will wait for a client to connect. Now, start a Tsuchinoko client in another shell:

```console
$ tsuchinoko
```
The client will automatically connect to the server. From the parameter table on the right, select an Acquisition Function to test (try starting with `gradient`).
To start the experiment, click the 'run' (▶) button.

```{image} _static/running-score.PNG
:alt: Tsuchinoko with no connection
:width: 500px
:align: center

```
&nbsp;

Now have fun!
- Experiment with different acquisition functions! Try switching between them while Tsuchinoko is running.
- Want to nudge Tsuchinoko in the right direction? Right click in the `Score` or `Posterior Mean` displays and select `Queue Measurement at Point`
- Save your work! You can save the current state of an experiment from File > Save..., or save an image of a display by right-clicking a display and selecting `Export...`.

```{image} _static/running-posterior-mean.PNG
:alt: Tsuchinoko with no connection
:width: 500px
:align: center

```
&nbsp;

## Next Steps

Now that you've seen Tsuchinoko in action, let's take a look at the server script that [describes the experiment](server_experiment.md).

(installation_troubleshooting)=
## Installation Troubleshooting

Some environments may need extra steps to install Tsuchinoko. Solutions are provided for these unusual cases below.

### libGL.so.1: cannot open shared object file: No such file or directory

You may be missing libgl on your system, for example if you are installing on a headless server. Look for a package 
that provides libgl in your package manager (i.e. for Ubuntu, install `libgl1-mesa-glx`)