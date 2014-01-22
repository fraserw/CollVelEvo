INC=-I /home/fraserw/NVIDIA_GPU_Computing_SDK/C/common/inc
INC2=-I /home/fraserw/NVIDIA_GPU_Computing_SDK/C/common/inc/cudpp
LIB=-L /home/fraserw/NVIDIA_GPU_Computing_SDK/cudpp_src_1.1.1/lib/

CollVelEvo : CollVelReadable_v1.o CollVelReadable_v1.cu random.o random.cu random.h reduce.o reduce.cu reduce.h routines.o routines.cu routines.h vel_evo.o vel_evo.cu vel_evo.h read_write.o read_write.h read_write.cu
	nvcc -o CollVelReadable --link CollVelReadable_v1.o routines.o reduce.o random.o vel_evo.o read_write.o -lcudpp $(INC) $(INC2) $(LIB)


reduce.o: reduce.cu reduce.h
	nvcc -arch sm_13 -c reduce.cu $(INC)

reduce6.o: reduce6.cu reduce6.h
	nvcc -o reduce6.o -c reduce6.cu $(INC)


routines.o : routines.cu routines.h
	nvcc -arch sm_13 -c routines.cu $(INC)

CollVelReadable_v1.o : CollVelReadable_v1.cu
	nvcc -arch sm_13 -c CollVelReadable_v1.cu $(INC)

random.o: random.h random.cu
	nvcc -arch sm_13 -c random.cu $(INC) $(INC2)

vel_evo.o: vel_evo.cu vel_evo.h
	nvcc -arch sm_13 -c vel_evo.cu 

read_write.o: read_write.h read_write.cu
	nvcc -c read_write.cu
clean : 
	rm *.o
	rm CollVelEvo
	#rm reduce-test 

