# Su Ming Yi
#  5441 lab4
# module load cuda/9.2.88
all: SuMingYi_lab4_v1.cu SuMingYi_lab4_v2.cu /fs/project/PAS1421/transform_cuda.o
	nvcc -dc SuMingYi_lab4_v1.cu
	nvcc -O -o lab4_arbitrary SuMingYi_lab4_v1.o /fs/project/PAS1421/transform_cuda.o
	nvcc -dc SuMingYi_lab4_v2.cu
	nvcc -O -o lab4_simd SuMingYi_lab4_v2.o /fs/project/PAS1421/transform_cuda.o
clean:
	rm -f lab4_arbitrary
	rm -f lab4_simd
	rm -f SuMingYi_lab4_v1.o
	rm -f SuMingYi_lab4_v2.o