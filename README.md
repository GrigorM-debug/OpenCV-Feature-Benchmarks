The repository contains Python scripts for testing Feature Extraction and Feature Matching algorithms.

Some time ago, I got interested in Simultaneous Localization and Mapping (SLAM) and started reading and watching lectures and other content about it, especially Monocular Visual SLAM. Feature Extraction and Feature Matching are part of Monocular Visual SLAM, also known as the front-end part.

There are a lot of algorithms you can use. They can be split into two groups:

**Feature Extraction**
- Traditional: SIFT, ORB, AKAZE
- Deep-Learning based: SuperPoint, DISK, ALIKED

**Feature Matching**
- Traditional: BF Matcher, FLANN
- Deep-Learning based: SuperGlue, LightGlue

I decided to test them and see how they perform. I made several tests on images and videos. I also made two benchmark tests for speed and accuracy.

# How to run the test

**Installation**
1. Clone the repo with the command below or download it as .ZIP 
```bash
   git clone https://github.com/GrigorM-debug/OpenCV-Feature-Benchmarks.git
```

2. Install the dependencies using 
```bash
   pip install -r requirements.txt
```

To use the Deep-Learning based methods, you need to install a few extra things. To do that, run the following commands:

1. Installing the algorithms 
```bash
   git clone https://github.com/cvg/LightGlue.git
   cd LightGlue
   python -m pip install -e .
   ```

Deep-Learning algorithms use PyTorch under the hood. When you run the commands above, PyTorch will be installed, but in my case it was a CPU-only version and when I ran the script I got an error. Yes, you can run the deep-learning methods on the CPU, but it will be slow. To fix the problem, I had to uninstall the installed version of PyTorch using:

```bash
pip uninstall torch torchvision torchaudio -y 
```

Then you have to check what version of CUDA your GPU supports using the following command:

```bash
nvidia-smi
```

After you know what CUDA version your GPU supports, you can install PyTorch from the official site: https://pytorch.org/get-started/locally/

## Tests on images

**Traditional methods**

First, you have to go to the directory containing the traditional algorithms using:

```bash
   cd traditional algorithms
```

Example of running one of the algorithms:
```bash
   python sift_bf_matcher.py "../images/image7.jpg" "../images/image8.jpg"
```

**Deep-Learning methods**

To run the deep-learning methods, you have to go to the directory using:

```bash
cd deep-leaning-methods
```

Then, you can run the scripts.
Example: 
```bash
python superpoint_and_lightglue.py "../images/image7.jpg" "../images/im
age8.jpg"
```

## Tests on videos
First, you have to go to the directory using: 
```bash
cd tests-on-videos
```

Then, you have to create a videos folder.
```bash
mkdir videos
```
In the videos folder, you have to put your own .mp4 videos.
I added my folder to `.gitignore` because the files are large.

**Tradional methods**
First, go to the directory.

```bash
cd tradional-methods
```

Example command to run them: 

```bash
python orb-and-bf.py "../videos/video.mp4"
```


**Deep-Learning methods**
Go to the directory using:
```bash
cd deep-learning-methods
```

Then run the script. Example command: 

```bash
python superpoint-and-lightglue.py "../videos/video2.mp4"
```

