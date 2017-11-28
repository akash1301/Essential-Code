/////////////////////// All Is Well /////////////////////////

#include <bits/stdc++.h>

#define FOR(i, s, e) for(int i=s; i<e; i++)
#define loop(i, n) for(int i=0; i<n; i++)
#define CIN   ios_base::sync_with_stdio(0); cin.tie(0)
#define getint(n) scanf("%d", &n)
#define pb(a) push_back(a)
#define ll long long int
#define ull unsigned long long int
#define dd double
#define SZ(a) int(a.size())
#define read() freopen("input.txt", "r", stdin)
#define write() freopen("output.txt", "w", stdout)
#define mem(a, v) memset(a, v, sizeof(a))
#define all(v) v.begin(), v.end()
#define pi acos(-1.0)
#define pf printf
#define sf scanf
#define mp make_pair
#define paii pair<int, int>
#define padd pair<dd, dd>
#define pall pair<ll, ll>
#define fr first
#define sc second
#define CASE(n) printf("Case %d:\n",++n)
#define CASE_COUT cout<<"Case "<<++cas<<": "
#define inf 1000000000
#define EPS 1e-9
#define Harmonic(n) (0.577215664901532+log(n)+(1/(2*n)))     ///Use Only for large n
#define mx 100005

using namespace std;

//8 way moves
//int fx[]={0,0,1,-1,1,1,-1,-1};
//int fy[]={1,-1,0,0,1,-1,1,-1};

//knight moves
//int fx[]={-2,-2,-1,-1,1,1,2,2};
//int fy[]={-1,1,-2,2,-2,2,-1,1};

//Bit operation
int SET(int n,int pos){ return n=n | (1<<pos);}
int RESET(int n,int pos){ return n=n & ~(1<<pos);}
int CHECK(int n,int pos){ return (bool) (n & (1<<pos));}

int str2int(string s) {
	stringstream ss(s);
	int x;
	ss >> x;
	return x;
}

string int2str(int a) {
	stringstream ss;
	ss << a;
	string str = ss.str();
	return str;
}

string char2str(char a) {
	stringstream ss;
	ss << a;
	string str = ss.str();
	return str;
}

ll bigMod(ll n,ll power,ll MOD)
{
    if(power==0)
        return 1;
    if(power%2==0)
    {
        ll ret=bigMod(n,power/2,MOD);
        return ((ret%MOD)*(ret%MOD))%MOD;
    }
    else return ((n%MOD)*(bigMod(n,power-1,MOD)%MOD))%MOD;
}

ll modInverse(ll n,ll MOD)
{
    return bigMod(n,MOD-2,MOD);
}

int POW(int x, int y)
{
    int res= 1;
    for ( ; y ; ) {
        if ( (y&1) ) {
            res*= x;
        }
        x*=x;
        y>>=1;
    }
    return res;
}

int inverse(int x)
{
    dd p=((dd)1.0)/x;
    return (p)+EPS;
}

int gcd(int a, int b)
{
    while(b) b^=a^=b^=a%=b;
    return a;
}

int nC2(int n)
{
    return n*(n-1)/2;
}

ll MOD(ll n,ll mod)
{
    if(n>=0)
        return n%mod;
    else if(-n==mod)
        return 0;
    else
        return mod+(n%mod);
}

int n,c,q,block,ans_max,cnt[mx],ans[mx],data[mx],cnt1[mx];
vector< pair< paii ,int> > point;

bool cmp(pair< paii ,int> xx,pair< paii ,int> yy)
{
    int pp=xx.fr.fr/block;
    int qq=yy.fr.fr/block;
    if(pp==qq) return xx.fr.sc<yy.fr.sc;
    return xx.fr.fr<yy.fr.fr;
}

void Add(int val)
{
    cnt1[cnt[val]]--;
    cnt[val]++;
    cnt1[cnt[val]]++;
    ans_max=max(ans_max,cnt[val]);
}

void Remove(int val)
{
    int pp=cnt1[cnt[val]];
    int qq=cnt[val];
    cnt1[cnt[val]]--;
    cnt[val]--;
    cnt1[cnt[val]]++;
    if(qq==ans_max && pp==1)
        ans_max--;
}


int main()
{
	int t,cas=0;
	getint(t);
	while(t--)
    {
        point.clear();
        sf("%d %d %d",&n,&c,&q);
        loop(i,n)
        {
            getint(data[i]);
        }
        loop(i,q)
        {
            int a,b;
            sf("%d %d",&a,&b);
            a--;
            b--;
            point.pb(mp(mp(a,b),i));
        }
        block=sqrt(n);
        sort(all(point),cmp);
        int lef=0,rt=-1;
        ans_max=0;
        mem(cnt,0);
        mem(cnt1,0);
        loop(i,q)
        {
            int x=point[i].fr.fr;
            int y=point[i].fr.sc;
            while(lef<x)
            {
                Remove(data[lef]);
                lef++;
            }
            while(lef>x)
            {
                lef--;
                Add(data[lef]);
            }
            while(rt<y)
            {
                rt++;
                Add(data[rt]);
            }
            while(rt>y)
            {
                Remove(data[rt]);
                rt--;
            }
            ans[point[i].sc]=ans_max;
        }
        CASE(cas);
        loop(i,q)
        {
            pf("%d\n",ans[i]);
        }
    }
	return  0;

}
