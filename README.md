# Panorama-Stitching
Robust Panoramic Image Stitching

Creation of panoramas using computer vision is not a new
idea, however, most algorithms are focused on creating a panorama
using all of the images in a directory. The method which will be
explained in detail takes this approach one step further by not
requiring the images in each panorama be separated out manually.
Instead, it clusters a set of pictures into separate panoramas based on
scale invariant feature matching. Then uses these separate clusters to
stitch together panoramic images.
