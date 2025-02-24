#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<omp.h>
//#include"file-reader.c" -- If you do it like this, make sure to remove file-reader.c function definitions just below

//Remove these if you do not do #include"file-reader.c"
int readNumOfPoints(char*);
int readNumOfFeatures(char*);
int readNumOfClasses(char *filename);
double *readDataPoints(char*, int, int);
void *writeResultsToFile(double*, int, char*);

int find_nearest(double*, double*, int, int, int, int, int, int*, double*, int);
void classify(int, double*, double*, int*, double*, int, int, int, int);

//Debugger function
void printArray(void *array, char array_name[500], int m, int n, char type);

int main(int argc, char *argv[]){

    printf("\n\n===============STARTING KNN===============\n\n");   

    //Load arguments 
    char *inFile = argv[1];
    char *outFile = argv[2];
    int k = atoi(argv[3]);
    int numFolds = atoi(argv[4]);

    //File reading
    int totalNumPoints, numFeatures, numClasses;
    double *originalData;
    
    totalNumPoints = readNumOfPoints(inFile);
    numFeatures = readNumOfFeatures(inFile);
    numClasses = readNumOfClasses(inFile);
    originalData = readDataPoints(inFile, totalNumPoints, numFeatures);

    //Array contains the number of points in each fold
    int *pointsInFold = (int *)malloc(numFolds * sizeof(int));
    double *accuracy = (double*)malloc((numFolds + 1) * sizeof(double));
    
    //Calculate the points in the fold and the remainder
    int pointsPerFold = totalNumPoints / numFolds;
    int pointsPerFoldRemainder = totalNumPoints % numFolds;

    //Allocates the number of points in the fold adding the remainder to the last fold. (Can handle unequally sized folds)
    for(int fold = 0; fold < numFolds; fold++){
        pointsInFold[fold] = pointsPerFold + ((fold == numFolds - 1) ? pointsPerFoldRemainder : 0);
    }

    /*Calculate the maximum size of the train and test folds
     * Since the last fold always has the different amount, we use this to our advantage*/
    size_t maxTrain_bytes = (totalNumPoints - pointsInFold[0]) * numFeatures * sizeof(double);
    size_t maxTest_bytes = pointsInFold[numFolds - 1] * numFeatures * sizeof(double);

    //Allocate the memory for the train and test datasets
    double *currTrain = (double*)malloc(maxTrain_bytes);
    double *currTest = (double*)malloc(maxTest_bytes);

    int *nearest = (int*)malloc(maxTest_bytes * k);
    double *nearestValues = (double*)malloc(maxTest_bytes * k);

    //Declare variables 
    size_t currTestSize, currTestSize_bytes, currTrainPoints, currTrainSize, currTrainSize_bytes, testOffset_bytes;

    //Test offset is used to track where we are in the originalData
    size_t testOffset = 0;

    double s_time = omp_get_wtime();

    for(int currFold = 0; currFold < numFolds; currFold++){
        
        currTestSize = pointsInFold[currFold] * numFeatures;
        currTestSize_bytes = currTestSize * sizeof(double);
        testOffset_bytes = testOffset * sizeof(double);

        //Copy the current fold into the currTest
        memcpy(currTest, originalData + testOffset, currTestSize_bytes);

        currTrainPoints = totalNumPoints - pointsInFold[currFold];
        currTrainSize = currTrainPoints * numFeatures;
        currTrainSize_bytes = currTrainSize * sizeof(double);

        //Copy the remaining folds into currTrain
        memcpy(currTrain, originalData, testOffset_bytes);
        memcpy(currTrain + testOffset, originalData + testOffset + currTestSize, currTrainSize_bytes - testOffset_bytes);

        //For debugging. (Good luck printing the asteroids dataset D: )
        
        //printArray(currTrain, "train", currTrainPoints, numFeatures, 'd');
        
        memset(nearest, -1, maxTest_bytes * k);
        
        find_nearest(currTrain, currTest, currTrainPoints, pointsInFold[currFold], numFeatures, numClasses, k, nearest, nearestValues, 0);
        
        //printArray(nearest, "nearest", pointsInFold[currFold], k, 'i');
        //printArray(nearestValues, "nearestValues", pointsInFold[currFold], k, 'd');

        classify(pointsInFold[currFold], originalData, currTest, nearest, nearestValues, numClasses, k, numFeatures, 0);

        //printArray(currTest, "test", pointsInFold[currFold], numFeatures, 'd');
        //printArray(originalData, "original", totalNumPoints, numFeatures, 'd');

        int matches = 0;
        for(int i = 0; i < pointsInFold[currFold]; i++){
            if(currTest[i * numFeatures + numFeatures - 1] == originalData[(int)currTest[i * numFeatures] * numFeatures + numFeatures - 1]){
                matches++;
            }
        }

        accuracy[currFold] = (double)matches / pointsInFold[currFold];

        
        testOffset += pointsInFold[currFold] * numFeatures;
    }

    

    double averageAccuracy = 0;

    for(int i = 0; i < numFolds; i++){
        printf("Fold %d: %d correct predictions made out of %d. Accuracy = %f\n\n", i, (int)(accuracy[i] * pointsInFold[i]), pointsInFold[i], accuracy[i]);
        averageAccuracy += accuracy[i];
    }  

    averageAccuracy = averageAccuracy / numFolds;

    accuracy[numFolds] = averageAccuracy;

    writeResultsToFile(accuracy, numFolds+1, outFile);

    printf("Average accuracy of program: %f\n\n", averageAccuracy);

    double e_time = omp_get_wtime();
    double time_ms = (e_time - s_time) * 1000;
    printf("Time taken: %f", time_ms);

    free(accuracy);
    free(nearest);
    free(nearestValues);
    free(originalData);
    free(currTrain);
    free(currTest);
}

/**
 * Prints a given array
 * @param array: the array being given
 * @param array_name: the name of the array
 * @param m: the number of elements for the m dimension
 * @param n: the number of elements for the n dimension, for 1 dimensional array, set this to 1
 * @param type: the type of the array. e.g. 'd'=double 'i'=integer.
 **/
void printArray(void *array, char array_name[500], int m, int n, char type){
    printf("\n\n====Printing %s=====\n\n", array_name);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(type == 'd') printf("%f, ", ((double*)array)[i * n + j]);
            else if (type == 'i') printf("%f, ", (double)((int*)array)[i * n + j]);

        }
        printf("\n");
    }
}