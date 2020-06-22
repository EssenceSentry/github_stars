# Bolton
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/rstreet85/bolton/master/LICENSE.txt) [![Build Status](https://travis-ci.org/rstreet85/bolton.svg?branch=master)](https://travis-ci.org/rstreet85/bolton) [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Java%20library%20to%20detect%20human%20skin%20in%20digital%20images%20&url=https://www.github.com/rstreet85/bolton&via=v0xstreet&hashtags=ComputerVision,HumanDetection,SkinDetection)

A standalone skin detection & segmentation program written in Java.

It's intended function is to accept a digital image, identify all pixels representing human skin, and remove everything else from the picture, creating a new image with only human skin displayed (in the future the opposite may be added as an option). The goal of this project is to eventually allow the user to choose from multiple published methods as well as directly set their parameters, so that anyone with a Computer Vision-based project requiring human detection may compare and find the best algorithm/parameters for their needs.

## Usage
**Command-line arguments:** *-filename* *-detectionMethod*

## Example
```
test.jpg explicityes
```
Original Image:
![Original Image](https://github.com/rstreet85/bolton/blob/master/test/test1.jpg)
Skinned Image:
![Skinned Image](https://github.com/rstreet85/bolton/blob/master/test/test1_out.png)

## Methods

Type | Method | Option
-----|--------|-------
Explicit | YES colorspace | *explicityes*

## :warning: Limitations :warning:
Explicit mode offers little distinction between skin and skin-like colors. More advanced methods involving training images and machine learning show more promise.

Original Image:
![Original Image](https://github.com/rstreet85/bolton/blob/master/test/test2.jpg)
Skinned Image:
![Skinned Image](https://github.com/rstreet85/bolton/blob/master/test/test2_out.png)

## To-Do
- Add more image formats for output besides .PNG; offer user choice
- Add preprocessing options:
  - Gaussian blurring
  - White balance correction
- Add probability map options