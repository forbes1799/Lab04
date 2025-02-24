#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<math.h>
#include<omp.h>
//#include"file-reader.c" -- If you include file-reader.c like this, make sure to remove file-reader.c function definitions just below and do not 

//Remove these if you do #include"file-reader.c"
int readNumOfPoints(char*);
int readNumOfFeatures(char*);
int readNumOfClasses(char *filename);
double *readDataPoints(char*, int, int);
void *writeResultsToFile(double*, int, char*);

int find_nearest(double*, double*, int, int, int, int, int, int*, double*, int);
void classify(int, double*, double*, int*, double*, int, int, int, int);

//Debugging
void printArray(void *array, char array_name[500], int m, int n, char type);

int main(int argc, char *argv[]){

    int commSz;
    int rank;
    int root = 0;

    //Initialise MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double s_time = MPI_Wtime();

    if(rank==root) printf("\n\n===============STARTING KNN===============\n\n");   

    //Load arguments 
    char *inFile = argv[1];
    char *outFile = argv[2];
    int k = atoi(argv[3]);
    int numFolds = atoi(argv[4]);

    //File reading
    int totalNumPoints, numFeatures, numClasses;
    double *originalData;

    int *pointsInFold = (int *)malloc(numFolds * sizeof(int));
    double *accuracy = (double*)malloc((numFolds + 1) * sizeof(double));

    //Only process 0 reads the data
    if(rank == root){
        totalNumPoints = readNumOfPoints(inFile);
        numFeatures = readNumOfFeatures(inFile);
        numClasses = readNumOfClasses(inFile);
        originalData = readDataPoints(inFile, totalNumPoints, numFeatures);

        int pointsPerFold = totalNumPoints / numFolds;
        int pointsPerFoldRemainder = totalNumPoints % numFolds; 

        int cumulative_points = 0;

        //Allocates the number of points in the fold adding the remainder to the last fold. (Can handle unequally sized folds)
        for(int fold = 0; fold < numFolds; fold++){
            pointsInFold[fold] = pointsPerFold + ((fold == numFolds - 1) ? pointsPerFoldRemainder : 0);
        }
    }

    //Broadcasting file data and file data to all other processes
    MPI_Bcast(&totalNumPoints, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&numFeatures, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&numClasses, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(pointsInFold, numFolds, MPI_INT, root, MPI_COMM_WORLD);

    //Allocate memory for originalData for ranks that did not read the data
    if(root != rank){
        originalData = (double*)malloc(totalNumPoints * numFeatures * sizeof(double));
    }

    //Broadcast the original data to all threads
    MPI_Bcast(originalData, totalNumPoints * numFeatures, MPI_DOUBLE, root, MPI_COMM_WORLD);

    /*Calculate the maximum size of the train and test folds
     * Since the last fold always has the different amount, we use this to our advantage*/
    size_t maxTrain_bytes = (totalNumPoints - pointsInFold[0]) * numFeatures * sizeof(double);
    size_t maxTest_bytes = pointsInFold[numFolds - 1] * numFeatures * sizeof(double);

    //Allocate the memory for the train and test datasets
    double *currTrain = (double*)malloc(maxTrain_bytes);
    double *currTest = (double*)malloc(maxTest_bytes);

    //Declare variables 
    size_t currTestSize, currTestSize_bytes, currTrainPoints, currTrainSize, currTrainSize_bytes, testOffset_bytes;

    //Test offset is used to track where we are in the originalData
    size_t testOffset = 0;

    int foldsPerProc;
    int foldsPerProcRemainder;

    int *foldsInProc = (int*)malloc(commSz * sizeof(int));
    int *sendcnts = (int*)malloc(commSz * sizeof(int));
    int *displs = (int*)malloc(commSz * sizeof(int));

    for(int currFold = 0; currFold < numFolds; currFold++){
        //Declare the currTrain folds
        currTrainPoints = totalNumPoints - pointsInFold[currFold];
        currTrainSize = currTrainPoints * numFeatures;
        currTrainSize_bytes = currTrainSize * sizeof(double);
        
        //Declare the currTest fold
        currTestSize = pointsInFold[currFold] * numFeatures;
        currTestSize_bytes = currTestSize * sizeof(double);
        testOffset_bytes = testOffset * sizeof(double);

        foldsPerProc = pointsInFold[currFold] / commSz;
        foldsPerProcRemainder = pointsInFold[currFold] % commSz;

        //Only process 0 splits the train and test
        if(rank == root){

            //Copy the current fold into the currTest
            memcpy(currTest, originalData + testOffset, currTestSize_bytes);

            //Copy the remaining folds into currTrain
            memcpy(currTrain, originalData, testOffset_bytes);
            memcpy(currTrain + testOffset, originalData + testOffset + currTestSize, currTrainSize_bytes - testOffset_bytes);

            for(int proc = 0; proc < commSz; proc++){
                foldsInProc[proc] = (foldsPerProc + ((proc == commSz - 1) ? foldsPerProcRemainder : 0));
                sendcnts[proc] = foldsInProc[proc] * numFeatures;
                displs[proc] = proc * foldsPerProc * numFeatures;
            }
        }

        MPI_Bcast(foldsInProc, commSz, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(sendcnts, commSz, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(displs, commSz, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(currTrain, currTrainSize, MPI_DOUBLE, root, MPI_COMM_WORLD);

        

        double *localCurrTest = (double*)malloc(sendcnts[rank] * sizeof(double));

        int *localNearest = (int*)malloc(sendcnts[rank] * k * sizeof(int));
        double *localNearestValues = (double*)malloc(sendcnts[rank] * k * sizeof(double));

        

        MPI_Scatterv(currTest, sendcnts, displs, 
                     MPI_DOUBLE, localCurrTest, 
                     sendcnts[rank], MPI_DOUBLE, 
                     root, MPI_COMM_WORLD);

        find_nearest(currTrain, localCurrTest, 
                     currTrainPoints, foldsInProc[rank], 
                     numFeatures, numClasses, k, 
                     localNearest, localNearestValues, rank);

        

        classify(foldsInProc[rank], originalData, localCurrTest, localNearest, localNearestValues, numClasses, k, numFeatures, 0);

        

        int totalMatches;
        int myMatches = 0; 
        for(int i = 0; i < foldsInProc[rank]; i++){
            if(localCurrTest[i * numFeatures + numFeatures - 1] == originalData[(int)(localCurrTest[i * numFeatures] * numFeatures + numFeatures - 1)]){
                myMatches++;
            }
        }

        MPI_Reduce(&myMatches, &totalMatches, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        if(rank == root){
            accuracy[currFold] = (double) totalMatches / pointsInFold[currFold];
        }
        
        free(localCurrTest);
        free(localNearest);
        free(localNearestValues);

        testOffset += pointsInFold[currFold] * numFeatures;
    }

    double accuracy_avg = 0.0;

    if(rank == root){
        for(int i = 0; i < numFolds; i++){
            accuracy_avg += accuracy[i];
        }
         accuracy_avg /= numFolds;

        accuracy[numFolds] = accuracy_avg;
        writeResultsToFile(accuracy, numFolds + 1, outFile);
    }

    if(rank==root)        free(accuracy);
    if(pointsInFold)      free(pointsInFold); 
    if(originalData)      free(originalData);
    if(currTest)          free(currTest);
    if(currTrain)         free(currTrain);

    double e_time = MPI_Wtime();

    if (rank==root) printf("\n\n\n Program takes %f ms to run\n\n\n", (e_time - s_time) * 1000);

    MPI_Finalize();
}

/**
 * Prints a given array
 * @param array: the array being given
 * @param array_name: the name of the array
 * @param m: the number of elements for the m dimension
 * @param n: the number of elements for the n dimension
 * @param type: the type of the array. e.g. d=double i=integer.
 **/
void printArray(void *array, char array_name[500], int m, int n, char type){
    printf("\n\n====Printing %s=====\n\n", array_name);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(type == 'd') printf("%f, ", ((double*)array)[i * n + j]);
            else if (type == 'i') printf("%d, ", ((int*)array)[i * n + j]);

        }
        printf("\n");
    }
}
