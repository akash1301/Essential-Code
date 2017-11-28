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
#define Unique(x)  x.erase(unique(all(x)), x.end())
#define pi acos(-1.0)
#define pf printf
#define sf scanf
#define mp make_pair
#define paii pair<int, int>
#define padd pair<dd, dd>
#define pall pair<ll, ll>
#define fr first
#define sc second
#define CASE(n) printf("Case %d: ",++n)
#define CASE_COUT cout<<"Case "<<++cas<<": "
#define inf 1000000000
#define EPS 1e-9
#define Harmonic(n) (0.577215664901532+log(n)+(1/(2*n)))     ///Use Only for large n
#define mx 40005

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

// ll modInverse(ll n,ll MOD)
// {
//     return bigMod(n,MOD-2,MOD);
// }

ll modInverse(ll a, ll m)
{
    ll m0 = m, t, q;
    ll x0 = 0, x1 = 1;

    if (m == 1)
      return 0;

    while (a > 1)
    {
        // q is quotient
        q = a / m;

        t = m;

        // m is remainder now, process same as
        // Euclid's algo
        m = a % m, a = t;

        t = x0;

        x0 = x1 - q * x0;

        x1 = t;
    }

    // Make x1 positive
    if (x1 < 0)
       x1 += m0;

    return x1;
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


/// Given two node a and b find the number of distinct weight on the path from a to b



int n,m,data[mx],st[mx],ed[mx],tim=0,level[mx],parent[mx],table[mx][20],block,viss[mx],cnt[100005];
vector<int>g[mx],order;
vector< pair< paii , paii > >query;

int dfs2(int u,int par)
{
    st[u]=++tim;
    order.pb(u);
    loop(i,g[u].size())
    {
        int v=g[u][i];
        if(v==par) continue;
        dfs2(v,u);
    }
    order.pb(u);
    ed[u]=++tim;
}


int dfs(int u,int par,int lev)
{
    level[u]=lev;
    parent[u]=par;
    loop(i,g[u].size())
    {
        int v=g[u][i];
        if(par==v) continue;
        dfs(v,u,lev+1);
    }
}


void lca_init()
{
    mem(table,-1);
    for(int i=1; i<=n; i++)
        table[i][0]=parent[i];
    for(int j=1; (1<<j)<=n; j++)
    {
        for(int i=1; i<=n; i++)
        {
            if(table[i][j-1]!=-1)
                table[i][j]=table[table[i][j-1]][j-1];
        }
    }
}


int lca_query(int p,int q)
{
    if(level[p]<level[q])
        swap(p,q);
    int log=1;
    while(1)
    {
        int aaa=log+1;
        if((1<<aaa)>level[p]) break;
        log++;
    }
    for(int i=log; i>=0; i--)
    {
        if(level[p]-(1<<i)>=level[q])
            p=table[p][i];
    }
    if(p==q) return p;
    for(int i=log; i>=0; i--)
    {
        if(table[p][i]!=-1 && table[p][i]!=table[q][i])
        {
            p=table[p][i];
            q=table[q][i];
        }
    }
    return parent[p];
}


bool comp(pair< paii , paii >xx,pair< paii , paii >yy )
{
    int x=xx.fr.fr/block;
    int y=yy.fr.fr/block;
    if(x==y) return xx.fr.sc<yy.fr.sc;
    return xx.fr.fr<yy.fr.fr;
}

int ans,anss[100005];

void Add(int index)
{
    int node=order[index];
    int val=data[node];
    viss[node]++;
    cnt[val]++;
    if(viss[node]==2)
    {
        cnt[val]-=2;
        if(cnt[val]==0)
            ans--;
    }
    else
    {
        if(cnt[val]==1)
            ans++;
    }
//    cout<<"ADD"<<endl;
//    cout<<node<<" "<<viss[node]<<" "<<cnt[val]<<" "<<ans<<endl;
}


void Remove(int index)
{
    int node=order[index];
    int val=data[node];
    viss[node]--;
    if(viss[node]==1)
    {
        cnt[val]++;
        if(cnt[val]==1)
            ans++;
    }
    else
    {
        cnt[val]--;
        if(cnt[val]==0)
            ans--;
    }
//    cout<<"Remove"<<endl;
//    cout<<node<<" "<<viss[node]<<" "<<cnt[val]<<" "<<ans<<endl;
}

vector<  paii >temp;

void Compression()
{
    sort(all(temp));
    loop(i,temp.size())
    {
        if(i>0 && temp[i].fr==temp[i-1].fr)
            data[temp[i].sc]=data[temp[i-1].sc];
        else
            data[temp[i].sc]=i+1;
    }
}

int main()
{
	int t,cas=0;
	sf("%d %d",&n,&m);
	loop(i,n)
	{
	    getint(data[i+1]);
	    temp.pb(mp(data[i+1],i+1));
	}

	Compression();

	loop(i,n-1)
	{
	    int a,b;
	    sf("%d %d",&a,&b);
	    g[a].pb(b);
	    g[b].pb(a);
	}

	tim=0;
	dfs2(1,0);

	dfs(1,0,0);
    lca_init();

	loop(i,m)
	{
	    int a,b;
	    sf("%d %d",&a,&b);
	    int lca=lca_query(a,b);
	    if(lca==a)
        {
            query.pb(mp(mp(st[a],st[b]),mp(i,-1)));
        }
        else if(lca==b)
        {
            query.pb(mp(mp(st[b],st[a]),mp(i,-1)));
        }
        else
        {
            if(st[a]<st[b])
            {
                query.pb(mp(mp(ed[a],st[b]),mp(i,lca)));
            }
            else
            {
                query.pb(mp(mp(ed[b],st[a]),mp(i,lca)));
            }
        }
	}


	block=sqrt((int)order.size());
	sort(all(query),comp);

	int lef=1,rt=0;
	ans=0;
	loop(i,m)
	{
	    int x=query[i].fr.fr;
	    int y=query[i].fr.sc;
//	    cout<<order[x]<<" "<<order[y]<<endl;
	    while(lef<x)
        {
            Remove(lef-1);
            lef++;
        }
        while(lef>x)
        {
            lef--;
            Add(lef-1);
        }
        while(rt<y)
        {
            rt++;
            Add(rt-1);
        }
        while(rt>y)
        {
            Remove(rt-1);
            rt--;
        }
        int lca=query[i].sc.sc;
        int ans2=ans;
        if(lca!=-1)
        {
            if(cnt[data[lca]]==0)
                ans2++;
        }
        anss[query[i].sc.fr]=ans2;
	}

	loop(i,m)
	{
	    pf("%d\n",anss[i]);
	}

	return  0;

}
