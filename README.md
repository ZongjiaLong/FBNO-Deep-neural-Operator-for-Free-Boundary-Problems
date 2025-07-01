# FBNO-Deep-neural-Operator-for-Free-Boundary-Problems
This is the code of the Deep neural Operator for Free Boundary Problems

## Data

All the data is available on Zenodo at [data](https://doi.org/10.5281/zenodo.15779011). In fact, since I'm not very good at using other scientific drawing tools, all the images for the paper were done using python drawing and text annotation on draw.io.In other words, all the original images can be generated from data and code.

## Environment

This package can be cloned locally and used with the following requirements:

```python
conda env create -f environment.yml
```

However I don't really recommend this approach, as I've worked on other projects on this environment, so it contains a lot of unnecessary libraries.I would rather recommend to configure the environment directly in the following way:

```python
seaborn==0.13.2
torch==1.9.0
scipy==1.15.2
matplotlib==3.9.0
tqdm==4.66.5
PyYAML==5.4.1
python=3.11.9
torch==2.0.1+cu118
numpy==1.26.0
tensorboard==2.17.1# not really necessary
```



## License

This software is distributed with the MIT license which translates roughly that you can use it however you want and for whatever reason you want. All the information regarding support, copyright and the license can be found in the LICENSE file.

## Acknowledgement

I appreciate the following github repos a lot for their valuable code base and datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thuml/Latent-Spectral-Models

https://github.com/Extrality/AirfRANS
