#include <iostream>
#include <iomanip>
#include <algorithm>
using namespace std;

/* int a[100][100]{};
int main(){
    int n,m;
    cin>>n>>m;
    for(int i=0;i < n;i++)
    {
        for(int j=0;j<m;j++)
        {
            cin>>a[i][j];
        }
    }
    for(int ii = 0;ii < m;ii++){
        for(int jj=n-1;jj >= 0;jj--)
        {
            cout<<a[jj][ii]<<" ";
        }
        cout<<endl;
    }
    system("pause");
    return 0;
} */
/* int main(){
    ios::sync_with_stdio(false);
    return 0;
} */

/* for(int j=n-1;j >= 0;j--)
    {
        int ja = j-1;
         for(int ja = j; ja >= 0;ja--) 
        while(a[j]>a[ja]&&ja>=0){
            ja--;
        }
    } */

//int cnt = 1;
//int n;
/* void trans(int a[], int s, int e)
{

}
 */
/* int main(){
    ios::sync_with_stdio(false);
    cin>>n;
    int *a = new int[n];
    for(int i=0;i < n;i++){
        cin>>a[i];
    }
    int i=n-1;
    int j=i-1;
    int so = a[i];
    while(j >= 0)
    {      
        if(so<=a[j]){
            so = a[j];
            cnt++;
        }
        j--;
    }
    cout<<n-cnt<<endl;
    //system("pause");
    return 0;
} */

/* int main(){
    ios::sync_with_stdio(false);
    long long n;
    int k;
    cin>>n>>k;
    long long x = n;
    int cnt = 1;
    if(n!=1)
    {
        while((x+1)%k!=0){
            cnt++;
            n--;
            if(n > 2)
                x += n;
            else{
                cnt += k-1;    
                break; 
            }
        }
        cnt++;
    }
    else{
        cnt = k;
    }
    if(k==1)
        cnt = 1;
    cout<<cnt<<endl;
    system("pause");
    return 0;
} */
/* 

int rown;               //五子棋
char str[21][21];
int wcnt =0;
int bcnt =0;
int d[4][2] = { 0, 1
               ,1, 0
               ,1, 1
               ,1, -1
               };

void dp(int i,int j,int x){
    if(str[i][j]!='#'){
        for(int n=1;n<5;n++)
        {
            int nx = i + n*d[x][0];
            int ny = j + n*d[x][1];
            int ox = nx - n*d[x][0];
            int oy = ny - n*d[x][1];
            if(str[nx][ny]!=str[ox][oy]) return;
        }
        if(str[i][j]=='W')
            wcnt++;
        else
            bcnt++;
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin>>rown;
    for(int i=1;i <= rown;i++){
        for(int j=1;j <= rown;j++){
            cin>>str[i][j];
        }
    }
    for(int ix=0;ix < 4;ix++){
       for(int i=1;i <= rown;i++)
          for(int j=1;j <= rown;j++)
              dp(i,j,ix);
    }
    cout<<bcnt<<' '<<wcnt<<endl;
    system("pause");
    return 0;
}

 */
/* #define ll long long            //约瑟夫环
ll k;

int fun(ll n,ll m)
{
	if(m==k) 
           return 1;
	if(n+m<=k) 
           return fun(n,n+m)+1;
	ll r = 1 + (n-(k-m+1))/k;
	int x = (m+n)%k;
	if(x==0)
          x=k;
	return fun(n-r,x)+1;
}

int main()
{
	ll n;cin>>n>>k;
	cout<<fun(n,1);
    system("pause");
	return 0;
} */

/* 设dp[i][j]为模d的余数为j的i位数的方案数，
那么考虑每次在当前数的尾部添加数字，有dp[i+1][(j*10+k)%d] += dp[i][j]这一转移，
其中k为尾部添加的数字，由于不能使用0和3来构造x，那么枚举k的时候只要避开0和3即可。
最后dp[n][0]即答案。初始化为dp[0][0] = 1，从0到n-1枚举i，将dp[i][j]向后继状态转移即可推出dp[n][0]的值。 */
/* 
#define ll long long                    
const int mod = 1000000007;
ll dp[1001][1001];
int main()
{
    int n, d;
    cin>>n>>d;
	dp[0][0] = 1;
    int i = 0;
    while(i<n)
	{
        int j=0;
		while(j<d)
		{
			if(dp[i][j]!=0){
                for(int k=1;k<=9;k++)
                {
                    if(k!=3){
                        dp[i+1][(j*10+k)%d] += dp[i][j];
                        dp[i+1][(j*10+k)%d] %= mod;
                    }
                }
            }
            j++;
		}
        i++;
	}
	cout<<dp[n][0]<<endl;
 */
/*  ll x = 0;
    int a = 1;
    for(int i=0;i < n;i++)
    {   x += 9*a;
        a = 10*a;
    }
    int cnt = 0;ll xnt = d;
    while(xnt <= x && xnt%x==0)
    {
        if(xnt){
            cnt++;
            xnt += d;
        }
    }
    cout<<x<<endl; */
   // cout<<0x7fffffff<<endl;
/*     system("pause");
    return 0;
} */
/* int s(int ch,int m){
    int a=0;
    while(ch != 0){
        a += ch%m;
        ch = ch/m;
    }
    return a;
} 

int main()
{
    ios::sync_with_stdio(false);
    int a,b,c;
    cin>>a>>b>>c;
    int z = 0;
    int acn[1000]{};
    for(int i=1;i <= 81; i++)
    {
        int x = i;
        for(int j=1;j < a;j++)      //pow(x,a)
        {
            x *= i;
        }
        x = b*x + c;
        int nu = s(x,10);
        if(nu == i)
        {
            acn[z] = x; 
            z++;
        }
    }
    cout<<z<<endl;
    for(int j=0;j < z;j++)
        cout<<acn[j]<<endl;        
    system("pause");
    return 0;
}  */
#define maxnum 1000
int mc(int);
int dim[maxnum];
int ma[maxnum][maxnum];
int best[maxnum][maxnum];


int McM(int i, int j){
    if(ma[i][j] != -1)
        return ma[i][j];
    if(i==j){
        ma[i][j] = 0;
        return 0;
    }
    int ans, max = 0xafafaf;
    for(int k=i;k<j;k++)
    {
        ans =McM(i,k)+McM(k+1,j)+dim[i-1]*dim[k]*dim[j];
        if(ans < max){
            best[i][j] = k;
            max = ans;
        }
    }
    ma[i][j] = max;
    return max;
}



int main() {
    int i,mnum;
    int mx = 0;
    while(EOF!=scanf("%d",&mnum)){
        for(i = 0;i <= mnum;i++)
            scanf("%d",&dim[i]);
        mx += mc(mnum);
    }
    cout<<mx<<endl;
    return 0;		
}