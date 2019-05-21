# Parallel Computing

@Author: Su, Ming Yi <br />
@Date: 05/21/2019 <br />
@Library: pthread, OpenMP, CUDA, MPI <br />

@Background
The producer - consumer problem is a classic programming exercise in computer science. <br/>
It has both high practical benefit as well as a straight-forward implementation. <br/>
This problem is composed of two parts. </br>
First, a “producer” creates some sort of workload, storing instructions or intermediate work in a queue. <br/>
Second, a “consumer” reads this queue, performs the desired work and produces appropriate output. <br/>


I implement the producer-consumer problem in different parallel API.</br>
There are 6 directories. <br/>
	- lab1 (serial program) <br />
	- lab2 (pthread) <br />
	- lab3 (OpenMP) <br />
	- lab4 (CUDA) <br />
	- lab5 (MPI) <br />
	- Ouput <br />

## ScreenShot1 - Pthread vs serial program
!["ScreenShot1"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab2.PNG)


## ScreenShot2 - The reduction ratio by pthread
!["ScreenShot2"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab2_2.PNG)


## ScreenShot3 - OpenMP with different number of threads
!["ScreenShot3"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab3.PNG)

## ScreenShot4 - OpenMP vs pthread vs serial program
!["ScreenShot4"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab3_2.PNG)

## ScreenShot5 - CUDA vs serial program
!["ScreenShot5"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab4.PNG)

## ScreenShot6 - CUDA SIMD vs without SIMD
!["ScreenShot6"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab4_2.PNG)

## ScreenShot7 - MPI Architecture
!["ScreenShot7"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab5.PNG)

## ScreenShot8 - MPI result
!["ScreenShot8"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab5_2.PNG)

## ScreenShot8 - Compare all parallel API
!["ScreenShot8"](https://github.com/bettle123/ParallelComputing/blob/master/Output/lab5_3.PNG)
