#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <map>
#include "base.h"

float calculateDistance(float a,float b,float c,float d){
    return sqrt(pow(a-c,2)+pow(b-d,2));
}


void knnSerial(float* coords, float* newCoords, int* classes, int numClasses, int numSamples, int numNewSamples, int k) {
    /* TODO: Put your sequential code in this function */
    for(int i=numNewSamples;i<numNewSamples+numSamples;i++){
        vector<vector<int>> temp; 
        for(int j=0;numSamples;j++){
            temp.push_back({calculateDistance(coords[2*j],coords[2*j+1],newCoords[2*i],newCoords[2*i+1]),classes[j]});
        }
        sort(temp.begin(),temp.end,[](vector &a,vector b){
            return a[0]<b[0];
        }
        );
        vector<int> temp2(k,0);
        for(int i=0;i<k;i++){
            temp2[temp[i][1]]++;
        }
        sort(temp2.begin(),temp2.end());
        classes[numSamples+i]=temp2[k-1];
    }
}
