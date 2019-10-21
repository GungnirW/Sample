
/* /* 描述
老师布置给小华一个题目.
这道题是这样的,首先给出n个在横坐标上的点,然后连续的用半圆连接他们:首先连接第一个点与第二点(以第一个点和第二点作为半圆的直径).然后连接第二个第三个点.
直到第n个点.现在需要判定这些半圆是否相交了,在端点处相交不算半圆相交.
输入
输入的第一行包含一个整数T(1 ≤ T ≤ 10)表示有T组样例.
每组样例的第一行是一个整数n(1 ≤ n ≤ 1000).
接下来的一行输入有n个用空格隔开的不同的整数.a1,a2,...,an(-1000000 ≤ ai ≤ 1000000),(ai,0)表示第i个点在横坐标的位置.输出
对于每个输入文件,输出T行.
每行输出"yes"表示这些半圆有相交或者"no".样例输入
2
4
0 10 5 15
4
0 15 5 10样例输出
yes
no 

int main()
{
    int *arra;
    int n;
    cin>>n;
    arra = new int[n];
    for(int i=0;i < n;i++)
        cin>> arra[i];
    return 0;
} */


#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

/*bool primenum(int p){
    for(int i=2;i <= sqrt(p);i++)
        if(p%i == 0)
            return false;
    return true;
}
6-1
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
 */
/*for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            iplus = a[i]+a[j];
            if(iplus%k == 0)
                icount++; 
        }
    } */
/*
int main(){                             n个数，两个和为k的倍数
    ios::sync_with_stdio(false);
    int n,k,iplus;
    int icount = 0;
    cin>>n>>k;
    int *a = new int[n];
    int b[1000000]{0};
    for(int c=0;c<n;c++){
        cin>>a[c];
        b[(a[c]%k)]++;
    }
    icount += (b[0]*(b[0]-1)/2);
    for(int i = 1;i < (float)k/2;i++)
        icount += (b[i]*b[k-i]);
    if(k%2 == 0)
        icount += (b[k/2]*(b[k/2]-1)/2);
    cout<<icount<<endl;
    delete []a;
    system("pause");
    return 0;
    }
 */   






/*int  main(){
    ios::sync_with_stdio(false);
    int i_total = 0;
    int n,m;
    cin>>n>>m;
    int a[1000]{0};
    for(int i=0;i < n;i++){             //求所有绳子长之和
        cin>>a[i];
        i_total += a[i];
    }
    float f_average = (float)i_total/m; //m条绳子可能的最大值
    float f_length;                     //记录绳长
    int left = 0;
    int right = f_average/(float)0.01;  //将0.01作为单位，right为0.01尺度下最大绳长
    while(right > left+1){                    
        int tm = 0;//teprim"m"
        f_length = (right+left)/2*0.01;//二分法求最大绳长
        for(int j=0;j < n;j++){
            tm += a[j]/f_length;
        }
        if(tm >= m)
            left = (left+right)/2; 
        if(tm < m)
            right = (left+right)/2;
    }
    // f_length = (right+left)/2*0.01;
    cout<<setiosflags(ios::fixed)<<setprecision(2)<<left/100.0<<endl;
    system("pause");
    return 0;
}
int main(){
    ios::sync_with_stdio(false);
    int i_total = 0;
    int n,m;
    cin>>n>>m;
    int a[1000]{0};
    for(int i=0;i < n;i++){             //求所有绳子长之和
        cin>>a[i];
        i_total += a[i];
    }
    float left = 0;float right = (float)i_total/m; //m条绳子可能的最大值
    while(left < right - 0.001){
        int count = 0;
        float mid = (left+right)/2;
        for(int i=0;i < n;i++)
            count += a[i]/mid;
        if(count >= m)
            left = mid;
        else
            right = mid;
    }
    cout<<setiosflags(ios::fixed)<<setprecision(2)<<left<<endl;
    system("pause");
    return 0;
}
*/
/*                           6-3
int test357(long int n){
    int m3=0,m7=0;
    int lnumber = 0;
    m3 = log(n)/log(3);
    m7 = log(n)/log(7);
    lnumber = (m7+1)*(m7+2)*(m7+3)/6 - 1;
    for(int s = m7+1;s <= m3;s++){
        for(int i = s;i>=0;i--){
            for(int j = s-i;j>=0;j--){
                lnumber++;
                long int a = pow(3,i)*pow(5,j)*pow(7,s-i-j);
                if(a > n)
                    return (--lnumber);
            }
        }
    }
    return lnumber;

}

int main()
{
    ios::sync_with_stdio(false);
    long int n;
    cin>>n;
    cout<<test357(n)<<endl;
    system("pause");
    return 0;
}*/
/*
int distance1[1002]{0};             7-2
bool c(int d,int n,int m){ 
    int i = m;
    int prim= 1;int k = 2;
    while(k<=n){
        while(distance1[k]-distance1[prim] < d){
            i--;
            k++;
            if(i<0) return false;
            if(k>n){
                if(prim!= 1) return true;
                else return false;
            }
        }
        prim= k;
        k++;
    }
    return true;
}

int main(){
    int n,m;
    int a=0;
    cin>>n>>m;
    for(int p=2;p<=n;p++){
        cin>>a;
        distance1[p] = a+distance1[p-1];
    }
    int left=0;int right = 1000001;
    while(left<right){
        int mid = (left+right+1)/2;//               keypoint:+1
        if(c(mid,n,m))
            left = mid;
        else 
            right = mid-1;
    }
    cout<<left<<endl;
    system("pause");
    return 0;
}
*/
int count1 = 1;
int n;
void Puttinginto(int icase[],int l,int i){
    if(i<n){
        if(icase[i]<=l){
            count1++;
        }
        Puttinginto(icase,l-icase[i],i+1);
        Puttinginto(icase,l,i+1);
    }
}
int main(){
    int l;
    cin>>n>>l;
    int *icase = new int[6];
    for(int i=0;i<n;i++)
        cin>>icase[i];
    Puttinginto(icase,l,0);
    cout<<count1<<endl;
    system("pause");
    return 0;
}
