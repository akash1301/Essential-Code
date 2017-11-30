/////////////////////////// UNION FIND DISJOINT SET /////////////////////
#include<cstdio>
#include<sstream>
#include<cstdlib>
#include<cctype>
#include<cmath>
#include<algorithm>
#include<set>
#include<queue>
#include<stack>
#include<list>
#include<iostream>
#include<fstream>
#include<numeric>
#include<string>
#include<vector>
#include<cstring>
#include<map>
#include<iterator>

#define FOR(i, s, e) for(int i=s; i<e; i++)
#define loop(i, n) for(int i=0; i<n; i++)
#define getint(n) scanf("%d", &n)
#define pb(a) push_back(a)
#define ll long long int
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
#define getlong scanf("%lld",&n)
#define CASE(n) printf("Case %d: ",++n)
#define inf 1000000000

using namespace std;

int fx[]={0,0,1,-1,1,1,-1,-1};
int fy[]={1,-1,0,0,1,-1,1,-1};

map<string,int>ind;
int make[5005],ans,cont[5005];

int find(int a)
{
    if(make[a]==a)
        return a;
    return make[a]=find(make[a]);
}


int join(int u,int v)
{
    int x=find(u);
    int y=find(v);
    if(x!=y)
    {
        make[x]=y;
        cont[y]+=cont[x];
        ans=max(ans,cont[y]);
        n--;
    }
}

int main()
{
    //read();
    int n,e;
    while(cin>>n>>e)
    {
        if(!n&&!e)
            break;
        loop(i,n)
        make[i]=i;
        ans=1;
        int in=0;
        getchar();
        loop(i,n)
        {
            string s;
            getline(cin,s);
            ind[s]=in;
            cont[in]=1;
            in++;
        }
        loop(i,e)
        {
            string s1,s2;
            cin>>s1>>s2;
            join(ind[s1],ind[s2]);
        }
        cout<<ans<<" "<<n<<endl;
        ind.clear();
    }
    return 0;
}

//////////////////////////// BFS /////////////////////////
vector<int>G[100];
void bfs(int src)
{
    queue<int>Q;
    Q.push(src);
    int visited[100]={0},level[100];
    int parent[100];
    visited[src]=1 ;
    level[src]=0;
    while(!Q.empty())
    {
        int u=Q.front();
        for(int i=0;i<G[u].size();i++)
        {
            int v=G[u][i];
            if(!visited[v])
            {
                level[v]=level[u]+1 ;
                parent[v]=u;
                visited[v]=1 ;
                Q.push(v);
            }
        }
        Q.pop();
    }
    for(int i=1 ;i<=n;i++)
    printf("%d to %d distance %d",src,i,level[i]);
}

int main()
{
    for(int i=1;i<=13;i++)
    {
        int m,n;
        cin>>n>>m;
        G[n].push_back(m);
        G[m].push_back(n);
    }
    int l;
    cin>>l;
    bfs(l);
    return 0;
}

///////////////////////////// DFS //////////////////////////
int taken[50];
vector<int>g[50];
void dfs(int p)
{
    taken[p]=1;
    for(int i=0;i<(int)g[p].size();i++)
        if(!taken[g[p][i]])
            dfs(g[p][i]);
}
int main()
{
    int e,n;
    while(cin>>n)
    {
        cin>>e;
        for(int i=0;i<e;i++)
        {
            int t,q;
            cin>>t>>q;
            g[t].pb(q);
            g[p].pb(t); //bidirectional graph
        }
        mem(taken,0);
        int s;
        cin>>s;
        dfs(s);
       for(int i=0;i<50;i++)
        g[i].clear();
    }
    return 0;
}

/////////////////////////////// BIPARTITE GRAPH CHECK ////////////////////////////////
#include <cstdio>
#include <vector>
#include <queue>
using namespace std;

#define MAX 1001

int n, e;
int partition[MAX], visited[MAX];
vector< int > G[MAX];

bool is_bipartite() {
    int i, u, v, start;
    queue< int > Q;
    start = 1; // nodes labeled from 1
    Q.push(start);
    partition[start] = 1; // 1 left, 2 right
    visited[start] = 1; // set gray
    while(!Q.empty()) {
        u = Q.front(); Q.pop();
        for(i=0; i < G[u].size(); i++) {
            v = G[u][i];
            if(partition[u] == partition[v]) return false;
            if(visited[v] == 0) {
                visited[v] = 1;
                partition[v] = 3 - partition[u]; // alter 1 and 2
                Q.push(v);
            }
        }
    }
    return true;
}

int main() {
    int i, u, v;
    scanf("%d %d", &n, &e);
    for(i = 0; i < e; i++) {
        scanf("%d %d", &u, &v);
        G[u].push_back(v);
        G[v].push_back(u);
    }
    if(is_bipartite()) printf("Yes\n");
    else printf("No\n");
    return 0;
}

/////////////////////////// DIJKSTRA //////////////////////////
#define mx 100001
#define nx 1000001
#define inf 10000000000000000
using namespace std;

vector<ll>g[mx],cost[mx];
ll parent[mx];
struct node
{
    ll city,dist;
    node (ll a,ll b)
    {
        city=a;
        dist=b;
    }
    bool operator < (const node& p) const {
    return dist>p.dist;
    }
};

ll dijkstra(ll dis)
{
    ll dt[mx];
    mem(dt,inf);
    priority_queue<node>q;
    mem(parent,-1);
    q.push(node(1,0));
    dt[1]=0;
    while(!q.empty())
    {
        node t=q.top();
        q.pop();
        ll u=t.city;
        if(u==dis)
            return dt[dis];
        for(ll i=0;i<(ll)g[u].size();i++)
        {
            ll v=g[u][i];
            if(cost[u][i]+dt[u]<dt[v])
            {
                dt[v]=cost[u][i]+dt[u];
                q.push(node(v,dt[v]));
                parent[v]=u;
            }
        }
    }
    return -1;
}


int main()
{
    ll n,m;
    cin>>n>>m;
    for(ll i=0;i<m;i++)
    {
        ll p,q,r;
        cin>>p>>q>>r;
        g[p].pb(q);
        g[q].pb(p);
        cost[p].pb(r);
        cost[q].pb(r);
    }
    ll x=dijkstra(n);
    if(x==-1)
        cout<<"-1"<<endl;
    else
    {
        ll u=n;
        vector<ll>out;
        while(u!=-1)
        {
            out.pb(u);
            u=parent[u];
        }
        reverse(all(out));
        for(ll i=0;i<(ll)out.size();i++)
            cout<<out[i]<<" ";
        cout<<endl;
    }
    return 0;
}

////////////////////////// FLOYD WARSHALL ///////////////////
int main()
{
    freopen("input.txt", "r", stdin);
    int g[100][100],n,e,next[100][100];
    while(cin>>n>>e)
    {
        for(int i=0;i<100;i++){
            for(int j=0;j<100;j++)
            {
                if(i==j)
                    g[i][j]=0;
                else
                    g[i][j]=inf;
                next[i][j]=j;
            }
        }
        for(int i=1;i<=e;i++)
        {
            int p,q,w;
            cin>>p>>q>>w;
            g[p][q]=w;
            g[q][p]=w;
        }
        for(int k=1;k<=n;k++)
            for(int i=1;i<=n;i++){
            for(int j=1;j<=n;j++)
            {
            if(g[i][k]+g[k][j]<g[i][j])
            {
                g[i][j]=g[i][k]+g[k][j];
                g[j][i]=g[j][k]+g[k][i];
                next[i][j]=next[i][k];
            }
            }
            }
        int strt,en,t;
        cin>>strt>>en;
        cout<<g[strt][en]<<endl;
        vector<int>v;
        v.pb(strt);
        while(strt!=en)
        {
            strt=next[strt][en];
            v.pb(strt);
        }
        for(int i=0;i<v.size();i++)
            cout<<v[i]<<" ";
        cout<<endl;
        v.clear();
    }
    return 0;
}

////////////////////// TOPOLOGICAL SORT //////////////////////
vector<int>g[101],top;
int wet[101];

int main()
{
    int n,m;
    while(cin>>n>>m)
    {
        if(n==0&&m==0)
            break;
        mem(wet,0);
        for(int i=1;i<=m;i++)
        {
            int p,q;
            cin>>p>>q;
            g[p].pb(q);
            wet[q]++;
        }
        for(int i=1;i<=n;i++)
        {
            if(wet[i]==0)
                top.pb(i);
        }
        for(int i=0;i<top.size();i++)
        {
            int v=top[i];
            for(int j=0;j<g[v].size();j++)
            {
                int u=g[v][j];
                wet[u]--;
                if(wet[u]==0)
                    top.pb(u);
            }
        }
        for(int i=0;i<top.size();i++)
            cout<<top[i]<<" ";
        cout<<endl;
        for(int i=0;i<101;i++)
            g[i].clear();
        top.clear();
    }
    return 0;
}

////////////////////////// ARTICULATION POINT //////////////////////
vector<int>g[mx];
int vis[mx],artpoint[mx],used[mx],low[mx],degroot,dfstime,comp_point[mx],comp;

void findart(int u, int par)
{
    int i, v, child = 0;
    used[u] = 1;
    vis[u] = low[u] = ++dfstime;
    for(i = 0; i < g[u].size(); i++)
    {
        v = g[u][i];
        if(v == par) continue;
        if(used[v]) low[u] = min(low[u], vis[v]);
        else
        {
            child++;
            findart(v, u);
            low[u] = min(low[u], low[v]);
            if(low[v] >= vis[u]) artpoint[u] = 1;
        }
    }
    if(par == -1) artpoint[u] = (child > 1);
}

int main()
{
    read();
    int n;
    while(cin>>n)
    {
        int e;
        cin>>e;
        loop(i,e)
        {
            int p,q;
            cin>>p>>q;
            g[p].pb(q);
            g[q].pb(p);
        }
        mem(artpoint,0);
        mem(vis,0);
        degroot=0;
        dfstime=0;
        findart(1,-1);
        loop(i,n)
        {
            if(artpoint[i]==1)
                cout<<i<<" ";
        }
    }
    return 0;
}

////////////////////////// ARTICULATION BRIDGE //////////////////////
vector<int>g[mx];
int vis[mx],bak[mx],dis[mx],dfsnum;
set< pair<int,int> >Bridges;


void dfs(int u, int par)
{
    int i, v;
    vis[u] = 1;
    dis[u] = bak[u] = ++dfsnum;
    for(i = 0; i < g[u].size(); i++)
    {
        v = g[u][i];
        if(v == par) continue;
        if(vis[v]) bak[u] = min(bak[u], dis[v]);
        else
        {
            dfs(v, u);
            bak[u] = min(bak[u], bak[v]);
            if(bak[v] > dis[u])
            {
                Bridges.insert(make_pair(u,v));
                Bridges.insert(make_pair(v,u));
            }
        }
    }
}
int main()
{
    read();
    int n;
    while(cin>>n)
    {
        int e;
        cin>>e;
        loop(i,e)
        {
            int p,q;
            cin>>p>>q;
            g[p].pb(q);
            g[q].pb(p);
        }
        mem(artpoint,0);
        mem(vis,0);
        degroot=0;
        dfsnum=0;
        dfs(1);
        loop(i,Bridges.size())
        {
            cout<<Bridges[i].first<<" "<<Bridges[i].second<<endl;
        }
    }
    return 0;
}

/////////////////////////////  BELLMAN FORD /////////////////////////////////////
#define inf 20000
using namespace std;

int main()
{
   //freopen("input.txt", "r", stdin);
    int t,d[1010];
    vector<int>gu,gv,cost;
    cin>>t;
    while(t--)
    {
        int n,m;
        cin>>n>>m;
        loop(i,m)
        {
            int p,q,w;
            cin>>p>>q>>w;
            gu.pb(p);
            gv.pb(q);
            cost.pb(w);
        }
        mem(d,inf);
        d[0]=0;
        bool k=true;
        loop(i,n)
        {
            int p=0;
            loop(j,m)
            {
                int u=gu[j];
                int v=gv[j];
                if(d[v]>d[u]+cost[j])
                {
                    p=1;
                    if(i==n-1)
                        k=false;
                    d[v]=d[u]+cost[j];
                }
            }
            if(p==0)
                break;
        }
        if(k==false)
            pf("possible\n");
        else
            pf("not possible\n");
        gu.clear();
        gv.clear();
        cost.clear();
    }
    return 0;
}

//////////////////////////// SEGMENT TREE //////////////////////////////
#define mx 100005
using namespace std;

int a[mx],ara[mx*3];

void init(int node,int b,int e)
{
    if(b==e)
    {
        ara[node]=a[b];
        return ;
    }
    int mid=(b+e)/2;
    int left=node*2;
    int right=node*2+1;
    init(left,b,mid);
    init(right,mid+1,e);
    ara[node]=min(ara[left],ara[right]);
}

int query(int node,int b,int e,int i,int j)
{
    if(i>e||j<b)
        return mx;
    if(b>=i&&e<=j)
        return ara[node];
    int mid=(b+e)/2;
    int right=node*2+1;
    int left=node*2;
    int p1=query(left,b,mid,i,j);
    int p2=query(right,mid+1,e,i,j);
    return min(p1,p2);
}

int main()
{
    //read();
    int t,x=0;
    getint(t);
    while(t--)
    {
        int n,q;
        getint(n);
        getint(q);
        for(int i=1;i<=n;i++)
        getint(a[i]);
        init(1,1,n);
        pf("Case %d:\n",++x);
        loop(i,q)
        {
            int x,y;
            getint(x);
            getint(y);
            pf("%d\n",query(1,1,n,x,y));
        }
    }
    return 0;
}

/////////////////////////////// STRONGLY CONNECTED COMPONENT ///////////////////////
#include <bits/stdc++.h>
using namespace std;

int n,mapa[mx],vis[mx],scc_num[mx],scc,node,tot;
vector<int>g[mx],finish,graph_scc[mx],rev_g[mx];

int dfs1(int u)
{
    vis[u]=1;
    loop(i,g[u].size())
    {
        int v=g[u][i];
        if(!vis[v])
            dfs1(v);
    }
    finish.pb(u);
}


int dfs2(int u)
{
    vis[u]=2;
    scc_num[u]=scc;
    loop(i,rev_g[u].size())
    {
        int v=rev_g[u][i];
        if(vis[v]==1)
            dfs2(v);
    }
}

int dfs3(int u)
{
    vis[u]=1;
    tot++;
    if(graph_scc[u].size()>1)
        node=1;
    loop(i,graph_scc[u].size())
    {
        int v=graph_scc[u][i];
        if(!vis[v])
            dfs3(v);
    }
}

void clr()
{
    loop(i,mx)
    {
        g[i].clear();
        rev_g[i].clear();
        graph_scc[i].clear();
    }
    finish.clear();
    mem(mapa,0);
    mem(vis,0);
    mem(scc_num,0);
}

int main()
{
	int t,cas=0;
	getint(t);
	while(t--)
    {
        clr();
        getint(n);
        int cnt=2;
        mapa[0]=1;
        loop(i,n)
        {
            int x;
            getint(x);
            loop(j,x)
            {
                int a,b;
                sf("%d %d",&a,&b);
                if(!mapa[a])
                    mapa[a]=cnt++;
                if(!mapa[b])
                    mapa[b]=cnt++;
                g[mapa[a]].pb(mapa[b]);
                rev_g[mapa[b]].pb(mapa[a]);
            }
        }

        for(int i=1;i<cnt;i++)
        {
            if(!vis[i])
            {
                dfs1(i);
            }
        }
        scc=1;
        for(int i=finish.size()-1;i>=0;i--)
        {
            int u=finish[i];
            if(vis[u]==1)
            {
                dfs2(u);
                scc++;
            }
        }

//        cout<<scc-1<<endl;

        for(int i=1;i<cnt;i++)
        {
            loop(j,g[i].size())
            {
                int u=g[i][j];
                if(scc_num[i]!=scc_num[u])
                    graph_scc[scc_num[i]].pb(scc_num[u]);
            }
        }

        mem(vis,0);

        node=0;
        tot=1;
        CASE(cas);
        dfs3(scc_num[1]);
        //cout<<node<<" "<<tot<<endl;
        if(!node && tot==scc)
            pf("YES\n");
        else
            pf("NO\n");
    }
	return  0;

}

//////////////////////Fast Fibbonacci ////////////////////////////

///large input <=10^18
///Fast doubling Method
///F(2n) = F(n)[2*F(n+1) – F(n)]
///F(2n + 1) = F(n)2 + F(n+1)2

ll a,b,c,d,mod=10000007;

void fast_fib(ll ara[],ll n)
{
    if(n==0)
    {
        ara[0]=0;
        ara[1]=1;
        return;
    }
    fast_fib(ara,(n/2));
    a=ara[0];
    b=ara[1];
    c=2*b-a;
    if(c<0)
        c+=mod;
    c=(c*a)%mod;
    d=(a*a+b*b)%mod;
    if(n%2==0)
    {
        ara[0]=c;
        ara[1]=d;
    }
    else
    {
        ara[0]=d;
        ara[1]=c+d;
    }
}


int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        ll n;
        cin>>n;
        ll ara[2]={0};
        fast_fib(ara,n);
        pf("%lld\n",ara[0]);
    }

}


/////////////// KMP ///////////////////
const int mx=1000005;

char T[mx],P[mx];
int prefix[mx];


void prefix_calc(int n)
{
    int now=0;
    prefix[0]=0;
    for(int i=1;i<n;i++)
    {
        while(now>0 && P[i]!=P[now])
            now=prefix[now-1];
        if(P[i]==P[now])
            now++;
        prefix[i]=now;
    }
}


int KMP(int n,int m)
{
    int now=0,cnt=0;
    for(int i=0;i<n;i++)
    {
        while(now>0 && T[i]!=P[now])
            now=prefix[now-1];
        if(T[i]==P[now])
            now++;
        if(now==m)
        {
            now=prefix[now-1];
            cnt++;
        }
    }
    return cnt;
}


int main()
{
	int t,cas=0;
	getint(t);
	while(t--)
    {
        sf("%s %s",T,P);
        int sz1=strlen(T);
        int sz2=strlen(P);
        mem(prefix,0);
        prefix_calc(sz2);
        CASE(cas);
        pf("%d\n",KMP(sz1,sz2));

    }
	return  0;

}
///////////////////////////////Matrix Exponentiation///////////////////////////////
ll mod=10007;

struct Matrix
{
    ll mat[4][4];
    int row,col;
    Matrix()
    {
        mem(mat,0);
    }
    Matrix(int r,int c)
    {
        row=r;
        col=c;
        mem(mat,0);
    }
    Matrix operator * (const Matrix &P) const
    {
        Matrix temp(row,P.col);
        for(int i=0;i<temp.row;i++)
        {
            for(int j=0;j<temp.col;j++)
            {
                ll sum=0;
                for(int k=0;k<col;k++)
                {
                    sum+=((mat[i][k]%mod) * (P.mat[k][j]%mod))%mod;
                    sum%=mod;
                }
                temp.mat[i][j]=sum;
            }
        }
        return temp;
    }
    Matrix operator + (const Matrix &P)const
    {
        Matrix temp(row,col);
        for(int i=0;i<temp.row;i++)
        {
            for(int j=0;j<temp.col;j++)
            {
                temp.mat[i][j]=((mat[i][j]%mod) + (P.mat[i][j]%mod))%mod;
            }
        }
        return temp;
    }
    Matrix IdentityMat()
    {
        Matrix temp(row,col);
        for(int i=0;i<temp.row;i++)
            temp.mat[i][i]=1;
        return temp;
    }
    Matrix Expo(ll power)
    {
        Matrix temp=(*this);
        Matrix ret=(*this).IdentityMat();
        while(power)
        {
            if(power%2==1)
                ret=ret*temp;
            temp=temp*temp;
            power/=2;
        }
        return ret;
    }
    void show()
    {
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
                cout<<mat[i][j]<<" ";
            cout<<endl;
        }
    }
};

////////////////// Hash /////////////////////////////////
/// Hashing Bases & MOD
///           0123456789
#define Base1 10000019ull
#define Base2 10000079ull
#define Base3 10000103ull
#define MOD1  1000000007ull
#define MOD2  1000000009ull
#define MOD3  1000000021ull
#define MX    100005

ull B1[MX],B2[MX];

void Init(){
    B1[0] = B2[0] = 1;

    for (int i=1;i<MX;i++){
        B1[i] = B1[i-1]*Base1;
        B2[i] = B2[i-1]*Base2;
    }
}

struct Hash{
    pair<ull,ull> H[MX];
    int digit[MX];
    int L;

    Hash(){
        L = 0;
        H[0] = mp(0,0);
    }

    void Insert(char x){
        digit[++L] = x-'a'+1;

        H[L].fr = H[L-1].fr * Base1 + digit[L];
        H[L].sc = H[L-1].sc * Base2 + digit[L];
    }

    pair<ull,ull> SubStr(int l,int r){
        int len = r-l+1;

        pair<ull,ull> ans;

        ans.fr = H[r].fr - H[l-1].fr * B1[len];
        ans.sc = H[r].sc - H[l-1].sc * B2[len];

        return ans;
    }

    pair<ull,ull> Concate(pair<ull,ull> h,int l,int r){
        pair<ull,ull> x = SubStr(l,r);

        h.fr = h.fr * B1[r-l+1] + x.fr;
        h.sc = h.sc * B2[r-l+1] + x.sc;

        return h;
    }

    bool operator==(const Hash& p)const{
        return L == p.L && H[L] == p.H[p.L];
    }

    pair<ull,ull>& operator[] (int index){
        return H[index];
    }
};

//////////////// Gaussian Elimination //////////////////

dd mat[1005][1005],X[1005];
int n;

void GaussianElimination()
{
    for(int j=1;j<=n;j++)
    {
        for(int i=1;i<=n;i++)
        {
            if(i>j)
            {
                dd c=mat[i][j]/mat[j][j];
                for(int k=1;k<=n+1;k++)
                {
                    mat[i][k]=mat[i][k] - (c*mat[j][k]);
                }
            }
        }
    }
    X[n]=mat[n][n+1]/mat[n][n];
    for(int i=n-1;i>=1;i--)
    {
        dd sum=0;
        for(int j=i+1;j<=n;j++)
        {
            sum+=mat[i][j] * X[j];
        }
        X[i]=(mat[i][n+1] - sum)/mat[i][i];
    }
}


int main()
{
	int t,cas=0;
	getint(n);
	for(int i=1;i<=n;i++)
	{
	    for(int j=1;j<=n+1;j++)
	    {
	        sf("%lf",&mat[i][j]);
	    }
	}

	GaussianElimination();

	for(int i=1;i<=n;i++)
        pf("%lf\n",X[i]);

	return  0;

}

/////////////////////////Minimum Vertex Cover/////////////////
int n;
vector<int>g[mx];
int dp[mx][3];

int dfs(int u,int ff,int par)
{
    if(dp[u][ff]!=-1) return dp[u][ff];
    if(ff)
    {
        int ret=1;
        loop(i,g[u].size())
        {
            int v=g[u][i];
            if(v==par) continue;
            ret+=min(dfs(v,1,u),dfs(v,0,u));
        }
        return dp[u][ff]=ret;
    }
    else
    {
        int ret=0;
        loop(i,g[u].size())
        {
            int v=g[u][i];
            if(v==par) continue;
            ret+=dfs(v,1,u);
        }
        return dp[u][ff]=ret;
    }
}

int main()
{
	int t,cas=0;
	getint(n);
	loop(i,n-1)
	{
	    int p,q;
	    sf("%d %d",&p,&q);
	    g[p].pb(q);
	    g[q].pb(p);
	}
	mem(dp,-1);
	int ans=min(dfs(1,1,0),dfs(1,0,0));
	pf("%d\n",ans);
	return  0;

}


///////////////// LCA //////////////////////////////
/// probelm link:http://www.spoj.com/problems/QTREE2/


int n,level[mx],parent[mx],table[mx][20],dis[mx];
vector< paii >g[mx];

int dfs(int u,int par,int lev)
{
    level[u]=lev;
    parent[u]=par;
    loop(i,g[u].size())
    {
        int v=g[u][i].fr;
        if(par==v) continue;
        dis[v]=dis[u]+g[u][i].sc;
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

int func(int node,int l)  /// l level er node ta return kore
{
    int log=1;
    while(1)
    {
        int aaa=log+1;
        if((1<<aaa)>level[node]) break;
        log++;
    }
    for(int i=log; i>=0; i--)
    {
        if(level[node]-(1<<i)>=l)
            node=table[node][i];
    }
    return node;
}

int main()
{
    int t,cas=0;
    getint(t);
    while(t--)
    {
        loop(i,mx)
            g[i].clear();
        getint(n);
        loop(i,n-1)
        {
            int a,b,c;
            sf("%d %d %d",&a,&b,&c);
            g[a].pb(mp(b,c));
            g[b].pb(mp(a,c));
        }
        mem(dis,0);
        dis[1]=0;
        dfs(1,0,0);
        lca_init();
        while(1)
        {
            //char ss[10];
            string ss;
            int a,b;
            //getchar();
            //sf("%s",ss);
            cin>>ss;
            if(ss=="DONE") break;
            sf("%d %d",&a,&b);
            int lca=lca_query(a,b);
            if(ss=="DIST")
            {
                int ans_dis=dis[a]+dis[b]-2*dis[lca];
                pf("%d\n",ans_dis);
            }
            else
            {
                int c;
                getint(c);
                int lef=level[a]-level[lca]+1;
                int rig=level[b]-level[lca]+1;
                if(lef>=c)
                {
                    int node_level=level[a]-c+1;
                    pf("%d\n",func(a,node_level));
                }
                else
                {
                    int node_level=c-lef+1;
                    node_level+=level[lca]-1;
                    pf("%d\n",func(b,node_level));
                }
            }

        }
    }
    return  0;

}


///////////////// 1D Sparse Table ///////////////////////////
const int k = 16;
const int N = 1e5;
const int ZERO = 1e9 + 1; // min(ZERO, x) = min(x, ZERO) = x (for any x)

int table[N][k + 1]; // k + 1 because we need to access table[r][k]
int Arr[N];

int main()
{
    int n, L, R, q;
    cin >> n; // array size
    for(int i = 0; i < n; i++)
        cin >> Arr[i]; // between -10^9 and 10^9

    // build Sparse Table
    for(int i = 0; i < n; i++)
        table[i][0] = Arr[i];
    for(int j = 1; j <= k; j++) {
        for(int i = 0; i <= n - (1 << j); i++)
            table[i][j] = min(table[i][j - 1], table[i + (1 << (j - 1))][j - 1]);
    }

    cin >> q; // number of queries
    for(int i = 0; i < q; i++) {
        cin >> L >> R; // boundaries of next query, 0-indexed
        int answer = ZERO;
        for(int j = k; j >= 0; j--) {
            if(L + (1 << j) - 1 <= R) {
                answer = min(answer, table[L][j]);
                L += 1 << j; // instead of having L', we increment L directly
            }
        }
        cout << answer << endl;
    }
    return 0;
}

//////////////////////// Centroid decomposition //////////////////////////

int n,tot_node,cen_par[mx],child[mx],root_cen;
vector<int>g[mx];

int dfs(int u,int par)
{
    tot_node++;
    child[u]=1;
    loop(i,g[u].size())
    {
        int v=g[u][i];
        if(v==par) continue;
        dfs(v,u);
        child[u]+=child[v];
    }
}

int centroid(int u,int par)
{
    loop(i,g[u].size())
    {
        int v=g[u][i];
        if(v==par) continue;
        if(child[v]>tot_node/2) return centroid(v,u);
    }
    return u;
}

void decompose(int node,int par)
{
    loop(i,g[node].size())
    {
        int v=g[node][i];
        if(v==par) g[node].erase(g[node].begin()+i);
    }
    tot_node=0;
    dfs(node,par);
    int cen=centroid(node,par);
    cen_par[cen]=par;
    if(par==-1)
        root_cen=cen;
    loop(i,g[cen].size())
    {
        int v=g[cen][i];
        decompose(v,cen);
    }
    g[cen].clear();
}

int main()
{
	int t,cas=0;
	sf("%d",&n);
	loop(i,n-1)
	{
	    int a,b;
	    sf("%d %d",&a,&b);
	    g[a].pb(b);
	    g[b].pb(a);
	}

	decompose(1,-1);
    cout<<endl;
	for(int i=1;i<=n;i++)
        cout<<i<<" "<<cen_par[i]<<endl;

	return  0;

}

/////////////////////// Trie Array Implementation ///////////////////////////

#include <bits/stdc++.h>

#define pii              pair <int,int>
#define pll              pair <long long,long long>
#define sc               scanf
#define pf               printf
#define Pi               2*acos(0.0)
#define ms(a,b)          memset(a, b, sizeof(a))
#define pb(a)            push_back(a)
#define MP               make_pair
#define db               double
#define ll               long long
#define EPS              10E-10
#define ff               first
#define ss               second
#define sqr(x)           (x)*(x)
#define D(x)             cout<<#x " = "<<(x)<<endl
#define VI               vector <int>
#define DBG              pf("Hi\n")
#define MOD              1000000007
#define CIN              ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0)
#define SZ(a)            (int)a.size()
#define sf(a)            scanf("%d",&a)
#define sfl(a)           scanf("%lld",&a)
#define sff(a,b)         scanf("%d %d",&a,&b)
#define sffl(a,b)        scanf("%lld %lld",&a,&b)
#define sfff(a,b,c)      scanf("%d %d %d",&a,&b,&c)
#define sfffl(a,b,c)     scanf("%lld %lld %lld",&a,&b,&c)
#define stlloop(v)       for(__typeof(v.begin()) it=v.begin();it!=v.end();it++)
#define loop(i,n)        for(int i=0;i<n;i++)
#define loop1(i,n)       for(int i=1;i<=n;i++)
#define REP(i,a,b)       for(int i=a;i<b;i++)
#define RREP(i,a,b)      for(int i=a;i>=b;i--)
#define TEST_CASE(t)     for(int z=1;z<=t;z++)
#define PRINT_CASE       printf("Case %d: ",z)
#define CASE_PRINT       cout<<"Case "<<z<<": "
#define all(a)           a.begin(),a.end()
#define intlim           2147483648
#define infinity         (1<<28)
#define ull              unsigned long long
#define gcd(a, b)        __gcd(a, b)
#define lcm(a, b)        ((a)*((b)/gcd(a,b)))

using namespace std;


/*----------------------Graph Moves----------------*/
//const int fx[]={+1,-1,+0,+0};
//const int fy[]={+0,+0,+1,-1};
//const int fx[]={+0,+0,+1,-1,-1,+1,-1,+1};   // Kings Move
//const int fy[]={-1,+1,+0,+0,+1,+1,-1,-1};  // Kings Move
//const int fx[]={-2, -2, -1, -1,  1,  1,  2,  2};  // Knights Move
//const int fy[]={-1,  1, -2,  2, -2,  2, -1,  1}; // Knights Move
/*------------------------------------------------*/

/*-----------------------Bitmask------------------*/
//int Set(int N,int pos){return N=N | (1<<pos);}
//int reset(int N,int pos){return N= N & ~(1<<pos);}
//bool check(int N,int pos){return (bool)(N & (1<<pos));}
/*------------------------------------------------*/

int trie[1000005][26];
int isWord[1000005];

int root,total,valid_char=26;



void init()
{
    root =0;
    total=0;
    for(int i=0;i<valid_char;i++)
        trie[root][i]=-1;
}

void insert(string str)
{
    int now=root;
    for(int i=0;i<SZ(str);i++)
    {
        int id=str[i]-'0';
        if(trie[now][id]==-1)
        {
            trie[now][id]=++total;
            ms(trie[total],-1);
        }
        now=trie[now][id];
    }
    isWord[now]++;
}

void del(string str)
{
    int now=root;
    for(int i=0;i<SZ(str);i++)
    {
        now=trie[now][str[i]-'0'];
    }
    isWord[now]--;
}

int query(string str)
{
   int now=root;
   int ans=0;
   for(int i=0;i<SZ(str);i++)
   {
        now=trie[now][str[i]-'0'];
   }
   return isWord[now];
}

int main()
{

//    freopen("in.txt","r",stdin);
    ///freopen("out.txt","w",stdout);

    init();


    return 0;
}

////////////////////////// Catalan Number ///////////////////////////////
ll cat[1005];

void CatalanNumber()
{
    cat[0]=cat[1]=1;
    for(int i=2;i<=1000;i++)
    {
        for(int j=0;j<i;j++)
        {
            ll p=(cat[j] * cat[i-1-j])%mod;
            cat[i]=(cat[i] + p)%mod;
        }
    }
}

ll dp[50][50];
///Finds the number of binary search tree not lower than height h...dp[n][h]
void CatalanHeight()
{
    dp[0][0]=1;
    dp[1][1]=1;
    for(int i=2;i<=35;i++)
    {
        for(int h=1;h<=i;h++)
        {
            for(int j=1;j<=i;j++)
            {
                for(int k=0;k<=h-1;k++)
                {
                    dp[i][h]+=(dp[j-1][h-1] * dp[i-j][k]);
                    if(k==h-1)
                        continue;
                    dp[i][h]+=(dp[j-1][k] * dp[i-j][h-1]);
                }
            }
        }
    }
}

int main()
{
    CatalanHeight();
	int t,cas=0;

	int n,h;
	cin>>n>>h;

	ll ans=0;

	for(int i=h;i<=n;i++)
    {
        ans+=dp[n][i];
    }

    cout<<ans<<endl;


	return  0;

}


//////////////////////// Mobious function /////////////////////////

int mu[mx]; /// mu[x]=0, x is divisible by some perfect square(except 1)
            /// mu[x]=1, x is square free and contains even number of prime factors
            /// mu[x]=-1 x is square free and contains odd number of prime factors


void Mobious(int n)
{
    for(int i=1;i<=n;i++)
        mu[i]=1;
    for(int i=0;prime[i]*prime[i]<=n;i++)
    {
        int p=prime[i]*prime[i];
        for(int j=p;j<=n;j+=p)
        {
            mu[j]=0;
        }
    }

    for(int i=0;i<prime.size();i++)
    {
        for(int j=prime[i];j<=n;j+=prime[i])
            mu[j]*=-1;
    }
}

////////////////// Gaussian elimination 2 /////////////////////

///Function returns the rank(number of non zero rows) in echelon form of the matrix

int gauss()
{
      int  r = 0;
      int mm=n+1;
      for(int c = 0 ; c < mm-1 && r < n ; c ++ ) {
            int j = r;
            for(int i= j+1; i < n; i ++) if(mat[i][c]) { j = i; break; }
            if(mat[j][c] == 0 ) continue;
            swap(mat[j],mat[r]);
            ll s = modInverse(mat[r][c], k);
            for(int i = 0; i < mm ; i++ ) mat[r][i] = (mat[r][i] * s) % k;
            for(int i = 0; i < n ; i ++ ) if(i!=r) {
                  if(mat[i][c] == 0) continue;
                  ll t = mat[i][c];
                  for(int j= 0; j < mm; j ++ ) mat[i][j] = ((mat[i][j] - t * mat[r][j]) % k + k ) % k;
            }
            r++;
      }
      return (ll)r;
}
