# Describing an experiment

A Tsuchinoko experiment is described in 2 parts:
- An **adaptive** engine which chooses positions to measure
- An **execution** engine which measures positions

These two elements are required for Tsuchinoko's Core service, which communicates with the client.

Let's look at an example experiment.

## Example

For this section, we'll look at the server_demo.py script discussed in [Getting Started](quickstart.md). In this 
example, Tsuchinoko will adaptively sample data from the source image to create a reconstruction by simulated 
measurements.

Skipping the imports, first we'll load the source image data to be sampled from. In this case, our source data is an RGB
JPEG image, so we'll average over the color dimension to form a luminosity map.

```python
# Load data from a jpg image to be used as a luminosity map
image = np.flipud(np.asarray(Image.open(Path(__file__).parent/'sombrero_pug.jpg')))
luminosity = np.average(image, axis=2)
```

Next, we can build a function that does the 'sampling'. Since our simulation source here is discrete, it helps to 
interpolate between pixels so as to avoid sharpness across pixel edges. Each measurement must include:
- The measured position (in some cases it may not be exactly the target position)
- The measured value
- The variance associated with that measurement
- A `dict` with any additional measurement data not covered above

```python
# Bilinear sampling will be used to effectively smooth pixel edges in source data
def bilinear_sample(pos):
    measured_value = ndimage.map_coordinates(luminosity, [[pos[1]], [pos[0]]], order=1)[0]
    variance = 1
    return pos, measured_value, variance, {}
```
Note that the variance returned here is simply `1`. In this example with simulated measurements, there's no empirical
measure of variance.

A simple **execution** engine can be constructed from the above function.

```python
execution = SimpleEngine(measure_func=bilinear_sample)
```

Let's also construct an **adaptive** engine. The
featured adaptive engine in Tsuchinoko is [gpCAM](https://gpcam.readthedocs.io/en/latest/).

```python
# Define a gpCAM adaptive engine with initial parameters
adaptive = GPCAMInProcessEngine(dimensionality=2,
                                parameter_bounds=[(0, image.shape[1]),
                                                  (0, image.shape[0])],
                                hyperparameters=[255, 100, 100],
                                hyperparameter_bounds=[(0, 1e5),
                                                       (0, 1e5),
                                                       (0, 1e5)])
```

Here the `GPCAMInProcessEngine` is constructed with a set dimensionality, the domain bounds (in this case the image shape),
and some hyperparameter initial values and bounds (for more info on these see [gpCAM's docs](https://gpcam.readthedocs.io/en/latest/api/autonomous-experimenter.html#autonomousexperimenter)).

Next, let's construct the core. The ZMQCore is standard.

```python
# Construct a core server
core = ZMQCore()
core.set_adaptive_engine(adaptive)
core.set_execution_engine(execution)
```

The `core.main()` in the last section starts the core's main loop, and is required for the core server to run.

```
if __name__ == '__main__':
    # Start the core server
    core.main()
```

Only when python runs this script with `python server_demo.py` will the above part be executed.

## Expanding on this example

In the above example, `measure_func` (or `bilinear_sample`) is the critical piece from which you might start expanding.
By providing your own `measure_func`, you could make 'measurements' any way you'd like.

For more advanced usages, you may even subclass the `Engine` classes to customize their functionality, or customize the
`Core` to modify the experimental process.

Users of [Bluesky](https://blueskyproject.io/) should note that a `BlueskyInProcessEngine` may provide convenience.
