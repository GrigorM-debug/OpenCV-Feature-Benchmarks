The repository contains pyhton script for testing Feature Extraction and Feature Matching algorithms.

Some time ago i got interested in Simultaneous Localization and Mapping (SLAM) and started reading and watching leactions and stuff about it especially Monocular Visual SLAM. Feature Extraction and Feature Matching are part of Monocular Visual SLAM SLAM and there are different algorithms. I decided to try some of them and make comparison.

For feature extraction i tried SIFT, AKAZE and ORB. I tried to test SURF also but i found out that it was removed from OpenCV. Maybe because it is patented.

For feature matching i tried Brute-Force Matcher and FLANN matcher.

I made combinations between different feature extraction and feature matching algorithms so how they work together.
