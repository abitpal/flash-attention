# flash-attention
Replicated flash attention 2 

## Why? 
Couple reasons as to why I did this: 
* Simple implementations of flash attention on the web greatly sacrifice speed for readability. When you implement flash attention, you realize that the specific implementation of flash attention matters exponentially more than the overall idea of flash attention when it comes to execution time. Therefore, with this repo, I wanted to capture aspects of what can speed up kernels while trying to maintain readability.
* For my sake. I often struggle with writing optimized code that optimize beyound time and space complexities so I wanted to challenge myself in writing something that's truly fast (faster than pytorch).
* To learn cuda.

## Repository Overview
There are 3 branches in the repository. 
1) **main**: This branch is the simple and very readable implementation of flash attention that's very very very slow.
2) **flash-optimized**: This branch contains the majority of the optimizations that I made - this is much faster than the naive implementation and compares well to pyTorch.
3) **flash-optimized-2**: This branch contains a little more optimizations on top of flash-optimized.

   
## Optimizations
I want to focus this repo on explaining the implementation specific optimization of flash attention that can make it truly fast. However, [here](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) is a great resource that helped me understand flash attention beyound the paper itself. 
