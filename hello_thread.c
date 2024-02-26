#include <stdlib.h>
#include <pthread.h>

void* helloThread(void* thread_value) {
	printf("Hello Thread!\n");
	return NULL;
}

int main() {
	pthread_t ID; /*pthread_t is the data type used to uniquely identify a thread*/
	pthread_create(&ID, NULL, helloThread, NULL);
	pthread_exit(NULL); /*this function is called after a thread has completed and no longer required.*/
	return 0;
}

