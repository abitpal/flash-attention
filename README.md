# flash-attention
Replicated flash attention 2 

## Why? 
Couple reasons as to why I did this: 
* Simple implementations of flash attention on the web greatly sacrifice speed for readability. When you implement flash attention, you realize that the specific implementation of flash attention matters exponentially more than the overall idea of flash attention when it comes to execution time. Therefore, with this repo, I wanted to capture aspects of what can speed up kernels while trying to maintain readability.
  
