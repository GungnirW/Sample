#include<iostream>
#include<algorithm>
using namespace std;

bool primenum(int p){
    for(int i=2;i <= sqrt(p);i++)
        if(p%i == 0)
            return false;
    return true;
}
//6-1
bool bPrimenum(int p){  //判断输入数字是否为素数
    if(p == 2||p ==3)
        return true;
    if(p%6 != 1 && p%6 != 5)
        return false;
    int imas_p = sqrt(p);
    for(int i = 5;i <= imas_p;i+=6)
        if(p%i == 0||p%(i+2) == 0)
            return false;
    return true;
}
 
bool bProblem6_1(int a){    //立方和质数
    int imas_a = sqrt(a);
    int i = 2;
    int prim;
    int iArr[500] = {0};
    iArr[0] = 2;
    iArr[1] = 3;
    for(int n = 5;n < a/2;n++){//记录从2到a的素数到iArr
        if(bPrimenum(n)){
            iArr[i] = n;
            i++;
        }
    }
    if(a%2 == 0){
        for(int im = 1;im < i;im++){
            if(prim = a-2-iArr[im])
                if(prim!=iArr[im]&&bPrimenum(prim))
                    return true;
        }
        return false;
    }
    else{
        for(int is = 1;is < i;is++){
            for(int ip = is+1;ip < i;ip++){
                if(prim = a-iArr[is]-iArr[ip])
                    if(prim!=iArr[is]&&prim!=iArr[ip]&&bPrimenum(prim))
                        return true;
            }
        }
        return false;
    }
}

 int main(){
    int a;
    cin>>a;
    if(bPrimenum(a)){
        if(bProblem6_1(a))
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
    else
        cout<<"No"<<endl;
    system("pause"); 
    return 0;
 }

//6-1
bool bPrimenum(int p){  //判断输入数字是否为素数
    if(p == 2||p ==3)
        return true;
    if(p%6 != 1 && p%6 != 5)
        return false;
    int imas_p = sqrt(p);
    for(int i = 5;i <= imas_p;i+=6)
        if(p%i == 0||p%(i+2) == 0)
            return false;
    return true;
}
bool bProblem6_1(int a){    //立方和质数
    int imas_a = sqrt(a);
    int i = 2;
    int prim;
    int iArr[500] = {0};
    iArr[0] = 2;
    iArr[1] = 3;
    for(int n = 5;n < a/2;n++){//记录从2到a的素数到iArr
        if(bPrimenum(n)){
            iArr[i] = n;
            i++;
        }
    }
    if(a%2 == 0){
        for(int im = 1;im < i;im++){
            if(prim = a-2-iArr[im])
                if(prim!=iArr[im]&&bPrimenum(prim))
                    return true;
        }
        return false;
    }
    else{
        for(int is = 1;is < i;is++){
            for(int ip = is+1;ip < i;ip++){
                if(prim = a-iArr[is]-iArr[ip])
                    if(prim!=iArr[is]&&prim!=iArr[ip]&&bPrimenum(prim))
                        return true;
            }
        }
        return false;
    }
}

 int main(){
    int a;
    cin>>a;
    if(bPrimenum(a)){
        if(bProblem6_1(a))
            cout<<"Yes"<<endl;
        else
            cout<<"No"<<endl;
    }
    else
        cout<<"No"<<endl;
    //system("pause"); 
    return 0;
 }