# Data Preparation

2D data is expected to be in the following file structure:
```
scene0000_00/
|--color/
   |--[framenum].jpg
       ⋮
|--depth/
   |--[framenum].png   (16-bit pngs)
       ⋮
|--pose/
   |--[framenum].txt   (4x4 rigid transform as txt file)
       ⋮
|--label/    (if applicable)
   |--[framenum].png   (8-bit pngs)
       ⋮
scene0000_01/
⋮
```

`prepare_2d_data.py` processes ScanNet data to fit this structure.

Dependencies:
* `numpy`, `imageio`, `scikit-image`, `opencv`
* depends on the [sens file reader from ScanNet](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py); should be placed in the same directory
* if export_label_images flag is on:
    * depends on [ScanNet util](https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py); should be placed in the same directory
    * assumes that label images are unzipped as scene*/label*/*.png

Example usage:
```
python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images
```