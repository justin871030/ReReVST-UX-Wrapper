# Usability wrapper for ReReVST

Original project from

Wenjing Wang, Shuai Yang, Jizheng Xu, and Jiaying Liu. **"Consistent Video Style Transfer via Relaxation and Regularization"**, _IEEE Trans. on Image Processing (TIP)_, 2020. https://doi.org/10.1109/TIP.2020.3024018 (see [citing papers on Scholar Google](https://scholar.google.co.uk/scholar?cites=4735550302416573229&as_sdt=2005&sciodt=0,5&hl=en))

Project Website: https://daooshee.github.io/ReReVST/ - https://github.com/daooshee/ReReVST-Code

![](https://daooshee.github.io/ReReVST/compare_result_video.jpg)

## Petteri's notes

This repo is a "wrapper repo" basically making the code a bit easier to run, if you for example want to quickly make artistic video style transfer and don't want to wrestle with the code forever. 

* Added **virtual environment** to make the work _a bit more_ reproducible
* Removed the need to do the mp4 -> png conversion as it got a bit annoying. No the inputs are just whatever **video files that `imageio ffmpeg` supports**
* Original repository did not keep **audio** at all, which was annoying if you wanted to stylize videos with audio tracks. This now handled with `MoviePy`

### Pre-prerequisites to get this code working on your machine

* Install [Anaconda3](https://www.anaconda.com/products/individual#windows) ([installation instructions](https://docs.anaconda.com/anaconda/install/windows/))
* Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), if you are `pull`ing this repo (you could just download the zip as well if you really do not know what this is)
* GO to terminal / command window here and execute the commands from there

### [Clone](https://medium.com/@madebymade/github-for-dummies-96f753f96a59) this repository

```bash
git clone https://github.com/petteriTeikari/ReReVST-Code
cd ReReVST-Code
```

### Get the pretrained model

Download `style_net-TIP-final.pth` [~60MB] from the links provided by the authors of the original repo:

Links: [Google Drive](https://drive.google.com/drive/folders/1RSmjqZTon3QdxBUSjZ3siGIOwUc-Ycu8?usp=sharing), [Baidu Pan](https://pan.baidu.com/s/1Td30bukn2nc4zepmSDs1mA) [397w]

And place this `.pth` file to `test/Model`

### Virtual environment (on which you run the code)

If you do not know what this, you could check for example [Python Virtual Environments: A Primer
](https://realpython.com/python-virtual-environments-a-primer/) or [What Virtual Environments Are Good For
](https://realpython.com/lessons/what-virtual-environments-are-good-for/)

#### GPU (assuming you have NVIDIA's GPU and it is okay with [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive))

Here a ~2 GB Pytorch package is installed, so if your internet is poor (like you live in London), this might take some time

```
python3.6 -m venv rerevst_venv
source rerevst_venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Using the repository to style your image

In machine learning jargon, you are now "testing" your model (running inference) after you initially trained it with a bunch of examples. If you just to want to quickly style your videos, you do not care about the training part of the code.

`generate_real_video.py` styles your input video defined in `--input_video` based on the style in an image `--style_img`.

To test the code with the default values and examples given by the paper

```
python generate_real_video.py --style_img ../inputs/styles/3d_4.jpg --input_video ../inputs/video/scatman.mp4
```

### Batch Processing

You probably do not know that well how different style images in the end will be transferred to the video, so you can copy all possible images that you could want to test to a single folder (e.g. `inputs/styles/`), and do something else while you get the results.

You can batch stylize a bunch of videos with a single style:

```
python generate_real_video.py --style_img ../inputs/styles/3d_4.jpg --input_video_dir ../inputs/video/
```

Single video file with multiple styles:

```
python generate_real_video.py --style_img_dir ../inputs/styles/ --input_video ../inputs/video/scatman.mp4
```

Bunch of videos with multiple styles:

```
python generate_real_video.py --style_img_dir ../inputs/styles/ --input_video_dir ../inputs/video/
```

### Optional settings

Additionally there is an `interval` flag that you can increase to `16`, `32` if you run out of RAM (if you high-res or/and long videos). Depends on your laptop/desktop on how much lifting you can do.

Then, you can find stylized video in `./results/videos/`.

With these optional setting, the inference call would look like:

```
python generate_real_video.py --style_img ../inputs/styles/3d_4.jpg --input_video ../inputs/video/scatman.mp4 --interval 16
```

#### CPU Processing

If you run out of GPU memory, or you have some issues with CUDA, you can force the computations to be done with CPU, makes things a lot slower as well. 

On `i7-7700K CPU @ 4.20GHz` ~1.30 frames are stylized per second compared to ~30 frames per second with [`NVIDIA RTX 2070 Super 8GB`](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/), i.e. stylizing a ~9sec video takes 3 min 20 sec on CPU, and around ~9seconds on a GPU. So definitely get a machine with a GPU if you plan to do these a lot :)

```
python generate_real_video.py --style_img ../inputs/styles/huygens_multiphoton_embryo.png --input_video ../inputs/video/scatman.mp4 --force_on_CPU True
```

### Processing time

Depending on your hardware, the actual style transfer can happen around in real-time (NVIDIA 2070 Super), ~9sec video took ~9sec to process with the model building and global feature sharing taking some time too:

```
Opened style image "3d_4.jpg"
Opened input video "scatman.mp4" for style transfer (fps = 30.0)
Stylization Model built in 0:00:07.734979
Preparations finished in 0:00:03.043060!
Video style transferred in 0:00:08.868579
Prcessing as a whole took 0:00:19.874647
```

### On how to train the model with your custom data

See the original repository https://github.com/daooshee/ReReVST-Code for instructions

### Raising [an issue](https://www.stevejgordon.co.uk/working-on-your-first-github-issue)

This repo released pretty much as it is to make casual video style transfer a bit easier as many of the repos out there were so poorly documented and were tricky to get running

![An Issue](doc/raise_an_issue.jpg)
