# ON THE STRUCTURES OF REPRESENTATION FOR THE ROBUSTNESS OF SEMANTICSEGMENTATION TO INPUT CORRUPTION

[Charles Lehman](https://charlielehman.github.io/), Dogancan Temel, [Ghassan AlRegib](http://www.ghassanalregib.com)

--------

<p align="center">
<img src="https://github.com/charlieLehman/segmentation_corruption/blob/master/resources/scribe.png" alt="Semantic Segmentation">
</p>


## Abstract
Semantic segmentation is a scene understanding task  at the heart of safety-critical applications where robustness to corrupted inputs is essential.  
Implicit Background Estimation (IBE) has demonstrated to be a promising technique to improve the robustness to out-of-distribution inputs for semantic segmentation models for little to no cost.
In this paper, we provide analysis comparing the structures learned as a result of optimization objectives that use Softmax, IBE, and Sigmoid in order to improve understanding their relationship to robustness.
As a result of this analysis, we propose combining Sigmoid with IBE (SCrIBE) to improve robustness.
Finally, we demonstrate that SCrIBE exhibits superior segmentation performance aggregated across all corruptions and severity levels with a mIOU of 42.1 compared to both IBE 40.3 and the Softmax Baseline 37.5.


## Citation: 
If you have found our code useful, we kindly ask you to cite our work: 
```tex
@INPROCEEDINGS{Lehman2020,
author={C. Lehman and D. Temel and G. AIRegib},
booktitle={IEEE International Conference on Image Processing (ICIP)},
title={On the Structures of Representation for the Robustness of Semantic Segmentation to Input Corruption},
year={2020},}
```
