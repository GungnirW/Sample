#include <iostream>
using namespace std;

int Max(int a,int b){
    return a > b?a:b;
}

int MaxSequence3(int *iArr,int n){
    int imax = 0;
    for(int length =1;length <= n;length++){
        for(int i = 0;i < n-length+1;i++){
            int current = 0, negative = 0;
            for(int j = i;j < i+length;j++){
                if(iArr[j]<0)
                    negative++;
                current+=iArr[j];
            }
            if(negative == length)  current =0;
            if(imax < current)    imax = current;
        }
    } 
    return imax;
}
int  MaxSequence2(int *iArr,int n){
    int imax = 0;    
    int *current = new int[n];
    int *negative = new int[n];
    for(int length = 1;length <= n;length++){
        for(int i = 0;i <= n-length;i++){
            current[i]+=iArr[i+length-1];
            if(iArr[i+length-1]<0)
                negative[i]++;
            if(negative[i]==length) 
                current[i] = 0;
            if(imax < current[i])    
                imax = current[i];
        }
    }
    delete []current;
    delete []negative;
    return imax;
}


int main(){
    /* int a[] = {-1,-2,4,9,-5,-1,6,1,-3};
    int n = sizeof(a)/sizeof(a[0]); 
     
    cout<<MaxSequence2(a,n)<<endl;
    //cout<<a[2]<<endl; */
    int **c;
    c = new int*[3];
    for(int i=0;i<3;i++){
        c[i] = new int[2]{i,i+1};
    }
    system("pause");
    return 0;    
} 