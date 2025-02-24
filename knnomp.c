#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>

//=======this file's functions=======//
void printData(double*, int, int);
int createAdjacencyMatrix(double**, double**, double**, int, int, int);
int modifynearest_neighbours(double*, int*, double*, double, int, int, int, int);
double getDistance(int, int, double*, double*, int);
int find_nearest(double*, double*, int, int, int, int, int, int*, double*, int);
void classify(int, double*, double*, int*, double*, int, int, int, int);


int find_nearest(double *train, double *test, int numTrainPoints, int numTestPoints, int num_features, int numClasses, int k, int *nearest, double *nearest_values, int rank){

    const int lnum_features = num_features;

    //Reshape 1-D array
    double (*train_data)[lnum_features] = (double(*)[lnum_features]) train;
    double (*test_data)[lnum_features] = (double(*)[lnum_features]) test;

    //Final column
    int labelIndex = num_features - 1;

    //printf("Finding nearest neighbours:\n");

    if(nearest == NULL || nearest_values == NULL){
        return EXIT_FAILURE;
    }

    double dist;

    #pragma omp parallel shared(train, test, numTrainPoints, numTestPoints, num_features, labelIndex, numClasses, k, nearest, nearest_values, train_data, test_data) private(dist)
    {

        //Initialise nearest values
        #pragma omp for nowait schedule(static)
        for(int i = 0; i < numTestPoints; i++){
            for(int j = 0; j < k; j++){
                nearest_values[i * k + j] = __DBL_MAX__;
            }
        }

        //Find the nearest neighbours
        #pragma omp for nowait schedule(static)
        for(int i = 0; i < numTestPoints; i++){
            for(int j = 0; j < numTrainPoints; j++){
                dist = getDistance(i, j, train, test, num_features);
                modifynearest_neighbours(train, nearest, nearest_values, dist, i, j, k, num_features);
            }
        }
    }
}

void classify(int numTestPoints, double *train, double *test, int *nearest, double *nearestValues, int num_classes, int k, int num_features, int rank){
    //Count the nearest neighbour labels

    //printf("Rank %d: Classifying!\n", rank);

    int *classCounts;
    int labelIndex = num_features - 1;

    #pragma omp parallel for private(classCounts) schedule(static)
    for(int i = 0; i < numTestPoints; i++){
        classCounts = (int *)calloc(num_classes, sizeof(int));

        for(int j = 0; j < k; j++){
            classCounts[(int)train[nearest[i * k + j] * num_features + labelIndex]]++;
        }

        int mostCommon = -1;
        int highestClassCount = -1;
        int commonClassesCount = 0;

        for(int x = 0; x < num_classes; x++){
            if(classCounts[x] > highestClassCount){
                mostCommon = x;
                highestClassCount = classCounts[x];
                commonClassesCount = 1; //Reset count if a new highest is found
            }
            else if(classCounts[x] == highestClassCount){
                commonClassesCount++;   //Increase count if a new 
            } 
        }

        //If there is more than 1 class with the same highest count, then there is a tie, and we choose the nearest neighbour
        if(commonClassesCount > 1){
            test[i * num_features + labelIndex] = train[nearest[i * k] * num_features +labelIndex];
        }
        else{
            test[i * num_features + labelIndex] = (double)mostCommon;
        }

        free(classCounts);
    }
}

//Function shifts nearest neighbours to the right. Sorts the array as it inserts. Easy handling when counting
inline int modifynearest_neighbours(double* train, int *nearest_neighbours, double *nearest_vals, double dist, int testPoint, int trainPoint, int k, int num_features){
    int inserted = 0;
    int pos = 1;

    //Reshape 1-D array
    int (*nearest)[k] = (int(*)[k]) nearest_neighbours;
    double (*nearest_values)[k] = (double(*)[k]) nearest_vals;

    while(dist < nearest_values[testPoint][k - 1] && inserted == 0){ 
        if(k - pos == 0){
            nearest_values[testPoint][k - pos] = dist;
            nearest[testPoint][k - pos] = train[trainPoint * num_features];
            inserted = 1;
        }
        else if(dist < nearest_values[testPoint][k - pos - 1]){
            nearest_values[testPoint][k - pos] = nearest_values[testPoint][k - pos - 1];
            nearest[testPoint][k - pos] = nearest[testPoint][k - pos - 1];
            pos++;
        }
        else{
            nearest_values[testPoint][k - pos] = dist;
            nearest[testPoint][k - pos] = train[trainPoint * num_features];
            inserted = 1;
        }
    }
    return 0;
}

//Euclidean distance based on how many features there are. Square root not necessary (too slow).
inline double getDistance(int testPoint, int trainPoint, double *train, double *test, int num_features){
    double (*train_data)[num_features] = (double(*)[num_features]) train;
    double (*test_data)[num_features] = (double(*)[num_features]) test;
    double sum = 0.0;

    #pragma omp simd
    for(int i = 1; i < num_features - 1; i++){
        double diff = train_data[trainPoint][i] - test_data[testPoint][i];
        sum += diff * diff;
    }
    return sum;
}

//If you want to print, print
void printData(double *points, int max_i, int max_j){
    double (*data)[max_j] = (double(*)[max_j]) points;
    for(int i = 0; i < max_i; i++){
        for(int j = 0; j < max_j; j++){
            printf("%f,", data[i][j]);
        }
        printf("\n");
    }
}