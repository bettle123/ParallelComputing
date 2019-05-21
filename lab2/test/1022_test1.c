/*
Author: Su Ming Yi
Date: 10/22/2018
@1022_test1.c

Goal: C Code for fork() creation test
how to compile it (incorrect):
gcc 1022_test1.c -o test1 -lpthread
how to run it:
./test1

 * */


#include <stdio.h>
#include <stdlib.h>
#define NFORKS 50000

void do_nothing() {
int i;
i= 0;
}

int main(int argc, char *argv[]) {
int pid, j, status;

for (j=0; j<NFORKS; j++) {

  /*** error handling ***/
  if ((pid = fork()) < 0 ) {
    printf ("fork failed with error code= %d\n", pid);
    exit(0);
    }

  /*** this is the child of the fork ***/
  else if (pid ==0) {
    do_nothing();
    exit(0);
    }

  /*** this is the parent of the fork ***/
  else {
    waitpid(pid, status, 0);
    }
  }
}  
