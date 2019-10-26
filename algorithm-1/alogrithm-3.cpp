#include <iostream>
#include <iomanip>
#include <algorithm>   
#include <string> 
using namespace std;
using std::string;

/* int Ollie(long a,long b,int r){ //2-1
    long p = a/b;long q = a%b;
    if(q&&p==1){
        r^=1;
        Ollie(b,q,r);
    }
    else
        return r;
}


int main(){ // 2-2
    int c=0;long a,b;
    ios::sync_with_stdio(false);
    cin>>c;
    int *n = new int[c];
    for(int i=0;i<c;i++){
        cin>>a>>b;
        if(a<b){
            long x = a;
            a = b;
            b = x;
        }
        int r = 1;
        n[i] = Ollie(a,b,r);
    }
    for(int i1=0;i1<c;i1++){
        if(n[i1])
            cout<<"Stan wins"<<endl;
        else
            cout<<"Ollie wins"<<endl;
    }
    system("pause");
    return 0;
} */
/* typedef struct Calender{
    int year;
    int month;
    int day;
}CALENDER;

int main(){
    ios::sync_with_stdio(false);

    return 0;
} */

/* int main(){ //2-6
    ios::sync_with_stdio(false);
    long n = 1;
    cin>>n;
    if(n==1)
        cout<<"181818181818"<<endl;
    else{
        long i = 1;
        while(i<n){
            if(9*i>=n){
                cout<<"181818181818"<<endl;
                return 0;
            }
            i *= 18;
        }
        cout<<"ZBT"<<endl;
    }
    system("pause");
    return 0;
} */
/* bool adjoin(int *t,int n){      //2-3
    sort(t,t+3);
    if(t[2]-t[1]==1&&t[1]-t[0]==1)
        return true;
    else{
        if(t[0]==0&&t[2]+t[1]==2*n-3)
            return true;
    }
    return false;
}

int main(){
    ios::sync_with_stdio(false);
    int n = 0;
    cin>>n;
    /* int **a= new int*[n-2];
    for(int i=0;i<n-2;i++){
        a[i] = new int[3];
        for(int j=0;j<3;j++)
            cin>>a[i][j];
    } 
    int (*a)[3] = new int[n][3];
    for(int i=0;i<n-2;i++){
        for(int j=0;j<3;j++)
            cin>>a[i][j];
    }
    if(adjoin(a[0],n))
        cout<<"JMcat Win"<<endl;
    else{
        if(n%2==0)
            cout<<"JMcat Win"<<endl;
        else
            cout<<"PZ Win"<<endl;
    }
    system("pause");
    return 0;
} */

/* int main(){     //2-5
    ios::sync_with_stdio(false);
    int a,k;
    cin>>k;
    int n;
    int *bl = new int[k];
    for(int i=0;i<k;i++){
        cin>>n>>bl[i];
        for(int j=0;j<n;j++)
            cin>>a;
    }
    for(int ij=0;ij<k;ij++)
    {
        if(bl[ij])
            cout<<"lolanv"<<endl;
        else 
            cout<<"wind"<<endl;

    }
    system("pause");
    return 0;
} */

/* int main(){     //2-4
    ios::sync_with_stdio(false);
    int a[10][2]{};
    for(int i=0;i<10;i++)
        for(int j=0;j<2;j++)
            cin>>a[i][j];
    for(int ii=0;ii<10;ii++){
        if((a[ii][0]%5 == 2||a[ii][0]%5 == 3)
        &&(a[ii][1]%5 == 2||a[ii][1]%5 == 3))
            cout<<"Shadow"<<endl;
        else
            cout<<"Matrix67"<<endl;
    }
    system("pause");
    return 0;
} */

/* int main(){     //
    ios::sync_with_stdio(false);
    
    system("pause");
    return 0;
} */
/* int main(){     //2-4
//ios::sync_with_stdio(false);加上以后输入流发生了改变  
    string s[2];
    int iWinner[10]{};
    for(int i=0;i<10;i++){
        int istate = 0;
        for(int j=0;j<2;j++)
        {
            char iw;
            cin>>s[j];
            iw = s[j].back();
            if(iw=='2'||iw=='3'||iw=='7'||iw=='8')
                istate++;
        }
        iWinner[i] = istate;
    }
    for(int ii=0;ii<10;ii++){
        if(iWinner[ii]==2)
            cout<<"Shadow"<<endl;
        else
            cout<<"Matrix67"<<endl;
    }
    system("pause");
    return 0; 
}  */


const int maxn=1005;
int a[maxn][maxn];
int f[maxn][maxn];
int n;

int main()
{
    cin>>n;
    for(int i=1;i<=n;i++)
        for(int j=1;j<=i;j++)
        cin>>a[i][j];
    f[1][1]=a[1][1];//终点处就直接是该点时间
    for(int i=2;i<=n;i++)//一层一层往上推
    {
        for(int j=2;j<i;j++)//先求出从上一层推出来的最小值
            f[i][j]=min(f[i-1][j],f[i-1][j-1])+a[i][j];
        f[i][1]=min(f[i-1][1],f[i-1][i-1])+a[i][1];//特殊边界点处理
        f[i][i]=min(f[i-1][i-1],f[i-1][1])+a[i][i];//特殊边界点处理
        //同一层更新最优解
        for(int k=i-1;k>0;k--)//从右往左推 从右往左走的情况更新
            f[i][k]=min(f[i][k],f[i][k+1]+a[i][k]);
        f[i][i]=min(f[i][i],f[i][1]+a[i][i]);

        for(int l=2;l<=i;l++)//从左往右推 从左往右走的情况更新
            f[i][l]=min(f[i][l],f[i][l-1]+a[i][l]);
            f[i][1]=min(f[i][1],f[i][i]+a[i][1]);

        /*for(int k=i-1;k>0;k--)//再推一遍从右往左推 从右往左走的情况更新
            f[i][k]=min(f[i][k],f[i][k+1]+a[i][k]);
            f[i][i]=min(f[i][i],f[i][1]+a[i][i]);

        for(int l=2;l<=i;l++)//再推一遍从左往右推 从左往右走的情况更新
            f[i][l]=min(f[i][l],f[i][l-1]+a[i][l]);
                f[i][1]=min(f[i][1],f[i][i]+a[i][1]);*/
    }
    cout<<f[n][1]<<endl;
    system("pause");
}
